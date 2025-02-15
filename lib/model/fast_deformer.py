import torch
import torch.nn.functional as F
from torch import einsum


from lib.model.network import ImplicitNetwork
from lib.model.deformer import skinning
from lib.model.broyden import broyden
from lib.model.helpers import bmv, create_voxel_grid, expand_cond, mask_dict
from torch.utils.cpp_extension import load
import os

class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, opt, **kwargs):
        super().__init__()

        self.opt = opt

        self.lbs_network = ImplicitNetwork(**self.opt.lbs_network)
        print(self.lbs_network)

        self.disp_network = ImplicitNetwork(**self.opt.disp_network)
        print(self.disp_network)

        self.soft_blend = 20

        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19]
        # self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()
        # init_lefthand = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        # init_righthand = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
        # init_face = [22, 23, 24]
        # self.init_bones = [init_bones_body, init_lefthand, init_righthand, init_face]
        self.n_init = len(self.init_bones)
        self.grid_denorm = create_voxel_grid(16, 64, 64)
        # self.n_init = [len(init_bone) for init_bone in self.init_bones]
        # self.n_init_lefthand = len(self.init_lefthand)
        # self.n_init_righthand = len(self.init_righthand)
        # self.n_init_face = len(self.init_face)

    def forward(self, xd, cond, tfs, mask=None, eval_mode=False, spatial_feat=False):
        """Given deformed point return its caonical correspondence

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            mask (tensor): valid points that need computation. shape: [B, N]

        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """

        n_batch, n_point_input, _ = xd.shape

        if mask is None:
            mask = torch.ones((n_batch, n_point), device=xd.device, dtype=torch.bool)

        tfs = expand_cond(tfs, xd)[mask]

        cond = { key:expand_cond(cond[key], xd) for key in cond}
        cond = mask_dict(cond, mask)

        # if pts_w != None:
        #     for key in cond:
        #         cond[key] = cond[key][:,None].expand(-1, 5, -1).flatten(0, 1)
        # else:
        for key in cond:
            cond[key] = cond[key][:,None].expand(-1, self.n_init, -1).flatten(0, 1)
            
        xd = xd[mask]
        # if pts_w != None:
        #     pts_w = pts_w[mask]

        xc_init = self.__init(xd, tfs)

        n_init = xc_init.shape[1]

        tfs = tfs[:,None].expand(-1, n_init, -1, -1, -1).flatten(0, 1)
        
        xc_opt, others = self.__search(xd, xc_init, cond, tfs, eval_mode=eval_mode, spatial_feat=spatial_feat)

        n_point, n_init, n_dim = xc_init.shape

        if not eval_mode:
            # compute correction term for implicit differentiation during training

            # do not back-prop through broyden

            xc_opt = xc_opt.detach()

            # reshape to [B,?,D] for network query
            xc_opt = xc_opt.reshape((n_point * n_init, n_dim))

            xd_opt = self.__forward_skinning(xc_opt, cond, tfs, spatial_feat=spatial_feat)

            grad_inv = self.__gradient(xc_opt, cond, tfs, spatial_feat=spatial_feat).inverse()

            correction = xd_opt - xd_opt.detach()
            correction = einsum("nij,nj->ni", -grad_inv.detach(), correction)

            # trick for implicit diff with autodiff:
            # xc = xc_opt + 0 and xc' = correction'
            xc = xc_opt + correction

            # reshape back to [B,N,I,D]
            xc = xc.reshape(xc_init.shape)
        else:
            xc = xc_opt

        xc = self.__query_cano(xc.reshape((n_point * n_init, n_dim)), cond).reshape(xc_init.shape)

        mask_root_find = torch.zeros( (n_batch, n_point_input, n_init), device=xc.device, dtype=torch.bool)
        mask_root_find[mask, :] = others['valid_ids']
        others['valid_ids'] = mask_root_find

        xc_full = torch.ones( (n_batch, n_point_input, n_init, n_dim), device=xc.device)

        xc_full[mask, :] = xc

        return xc_full, others


    def query_cano(self, xc, cond, mask=None, val_pad=0):
        """Given canonical (with betas) point return its correspondence in the shape neutral space
        Batched wrapper of __query_cano

        Args:
            xc (tensor): canonical (with betas) point. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor): valid points that need compuation. shape: [B, N]

        Returns:
            xc (tensor): correspondence in the shape neutral space. shape: [B, N, I, D]
        """

        input_dim = xc.ndim
        if input_dim == 3:
            n_batch, n_point, n_dim = xc.shape

            if mask is None:
                mask = torch.ones( (n_batch, n_point), device=xc.device, dtype=torch.bool)

            cond = { key:expand_cond(cond[key], xc) for key in cond}
            cond = mask_dict(cond, mask)

            xc = xc[mask]
        out = self.__query_cano(xc, cond)

        if input_dim == 3:
            out_full = val_pad * torch.ones( (n_batch, n_point, out.shape[-1]), device=out.device, dtype=out.dtype)
            out_full[mask] = out
        else:
            out_full = out

        return out_full

    def query_weights(self, xc, cond, mask=None, val_pad=0, spatial_feat=False):
        """Get skinning weights in canonical (with betas) space. 
        Batched wrapper of __query_weights

        Args:
            xc (tensor): canonical (with betas) point. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor): valid points that need compuation. shape: [B, N]

        Returns:
            w (tensor): skinning weights. shape: [B, N, J]
        """

        input_dim = xc.ndim

        if not hasattr(self,"lbs_voxel_final"):
            self.mlp_to_voxel(device=xc.device)

        if input_dim == 3:
            n_batch, n_point, n_dim = xc.shape

            if mask is None:
                mask = torch.ones( (n_batch, n_point), device=xc.device, dtype=torch.bool)

        #     cond = { key:expand_cond(cond[key], xc) for key in cond}
        #     cond = mask_dict(cond, mask)

        #     xc = xc[mask]

        out = self.__query_weights(xc, cond, warp=False, spatial_feat=spatial_feat)

        if input_dim == 3:
        #     out_full = val_pad * torch.ones( (n_batch, n_point, out.shape[-1]), device=out.device, dtype=out.dtype)
        #     out_full[mask] = out
        # else:
        #     out_full = out
            out[~mask] = val_pad
        else:
            out = out.reshape(-1, 24)

        return out

    def __init(self, xd, tfs):
        """Transform xd to canonical space for initialization

        Args:
            xd (tensor): deformed points in batch. shape: [N, D]
            tfs (tensor): bone transformation matrices. shape: [N, J, D+1, D+1]

        Returns:
            xc_init (tensor): gradients. shape: [N, I, D]
        """
        n_point, n_dim = xd.shape
        n_point, n_joint, _, _ = tfs.shape

        xc_init = []

        # if pts_w != None:
        #     n_knn = pts_w.shape[1]
        #     for i in range(n_knn):
        #         w = pts_w[:, i, :]
        #         xc_init.append(skinning(xd, w, tfs, inverse=True))
        # else:
        for i in self.init_bones:
            w = torch.zeros((n_point, n_joint), device=xd.device)
            w[:, i] = 1
            xc_init.append(skinning(xd, w, tfs, inverse=True))

        xc_init = torch.stack(xc_init, dim=-2)

        return xc_init.reshape(n_point, len(self.init_bones), n_dim)
        # return xc_init.reshape(n_point, -1, n_dim)

    def __search(self, xd, xc_init, cond, tfs, eval_mode=False, spatial_feat=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [N, D]
            xc_init (tensor): deformed points in batch. shape: [N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [N, J, D+1, D+1]

        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [N, I, D]
            valid_ids (tensor): identifiers of converged points. [N, I]
        """
        if not eval_mode or not hasattr(self,"lbs_voxel_final"):
            self.mlp_to_voxel(device=tfs.device)
        # reshape to [B,?,D] for other functions
        n_point, n_init, n_dim = xc_init.shape
        xc_init = xc_init.reshape(n_point * n_init, n_dim)
        xd_tgt = xd[:,None].expand(-1, n_init, -1).flatten(0, 1)


        # compute init jacobians
        if not eval_mode:
            J_inv_init = self.__gradient(xc_init, cond, tfs, spatial_feat=spatial_feat).inverse()
        else:
            w = self.query_weights(xc_init, cond, spatial_feat=spatial_feat)
            J_inv_init = einsum("pn,pnij->pij", w, tfs)[:, :3, :3].inverse()

        # reshape init to [?,D,...] for boryden
        xc_init = xc_init.reshape(-1, n_dim, 1)

        # construct function for root finding
        def _func(xc_opt, mask, spatial_feat=False):
            # reshape to [B,?,D] for other functions

            xc_opt = xc_opt[mask].squeeze(-1) #.reshape(n_batch, n_point * n_init, n_dim)

            xd_opt = self.__forward_skinning(xc_opt, mask_dict(cond, mask), tfs[mask], spatial_feat=spatial_feat)

            error = xd_opt - xd_tgt[mask]

            # reshape to [?,D,1] for boryden
            error = error.unsqueeze(-1)
            return error

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, xc_init, J_inv_init, max_steps=self.opt.max_steps, spatial_feat=spatial_feat)

        # reshape back to [B,N,I,D]
        xc_opt = result["result"].reshape(n_point, n_init, n_dim)

        result["valid_ids"] = result["valid_ids"].reshape(n_point, n_init)

        return xc_opt, result

    def __forward_skinning(self, xc, cond, tfs, spatial_feat=False):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [N, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        w = self.query_weights(xc, cond, spatial_feat=spatial_feat)
        xd = skinning(xc, w, tfs, inverse=False)
        return xd


    def __query_cano(self, xc, cond):
        """Map point in canonical (with betas) space to shape neutral canonical space

        Args:
            xc (tensor): canonical points. shape: [N, D]
            cond (dict): conditional input.

        Returns:
            w (tensor): skinning weights. shape: [N, J]
        """
        return self.disp_network(xc, cond) + xc
            
    def __query_weights(self, xc, cond, warp=True, spatial_feat=False):
        """Get skinning weights in canonical (with betas) space

        Args:
            xc (tensor): canonical points. shape: [N, D]
            cond (dict): conditional input.

        Returns:
            w (tensor): skinning weights. shape: [N, J]
        """
        # if warp:
        #     xc = self.__query_cano(xc, cond)
        
        # w = self.lbs_network(xc, cond, spatial_feat=spatial_feat, lbs=True)

        # w = self.soft_blend * w

        # w = F.softmax(w, dim=-1)

        # return w

        if warp:
            xc = self.__query_cano(xc, cond)
        if xc.ndim == 2:
            xc = xc.unsqueeze(0)
        if xc.shape[0] != 1:
            w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0], -1, -1, -1, -1), xc.unsqueeze(2).unsqueeze(2), align_corners=self.opt.align_corners, mode='bilinear', padding_mode='zeros')
        else:
            w = F.grid_sample(self.lbs_voxel_final, xc.unsqueeze(2).unsqueeze(2), align_corners=self.opt.align_corners, mode='bilinear', padding_mode='zeros')
        w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
        
        return w

    def __gradient(self, xc, cond, tfs, spatial_feat=False):
        """Get gradients df/dx

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            grad (tensor): gradients. shape: [B, N, D, D]
        """
        xc.requires_grad_(True)

        xd = self.__forward_skinning(xc, cond, tfs, spatial_feat=spatial_feat)

        grads = []
        for i in range(xd.shape[-1]):
            d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
            d_out[..., i] = 1
            grad = torch.autograd.grad(
                outputs=xd,
                inputs=xc,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)

        return torch.stack(grads, dim=-2)
    
    def mlp_to_voxel(self, device):
        input_dim = self.grid_denorm.ndim
        n_batch, n_point, n_dim = self.grid_denorm.shape

        d, h, w = self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res

        lbs_voxel_final = self.lbs_network(self.grid_denorm.reshape(-1, n_dim).to(device), cond=None, mask=None)
        
        if input_dim == 3:
            lbs_voxel_final = lbs_voxel_final.reshape(n_batch, n_point, -1)

        lbs_voxel_final = self.opt.soft_blend * lbs_voxel_final
        lbs_voxel_final = F.softmax(lbs_voxel_final, dim=-1)
    
        self.lbs_voxel_final = lbs_voxel_final.permute(0,2,1).reshape(1,24,d,h,w)



def skinning(x, w, tfs, inverse=False, normal=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [N, D]
        w (tensor): conditional input. [N, J]
        tfs (tensor): bone transformation matrices. shape: [N, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [N, D]
    """

    if normal:
        tfs = tfs[:,:,:3,:3]
        w_tf = einsum('pn,pnij->pij', w, tfs)
        w_tf_invt = w_tf.inverse().transpose(-2,-1)
        x = einsum('pij,pj->pi', w_tf_invt, x)

        return x
        
    else:

        p_h = F.pad(x, (0, 1), value=1.0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            w_tf = einsum('pn,pnij->pij', w, tfs)
            p_h = einsum('pij,pj->pi', w_tf.inverse(), p_h)
        else:
            p_h = einsum('pn, pnij, pj->pi', w, tfs, p_h)
        return p_h[:, :3]
