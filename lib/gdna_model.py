import os

import hydra
import torch
import wandb
import imageio
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import itertools

import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import copy
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor
)
from lib.utils.render import Renderer
# import nvdiffrast.torch as dr

from lib.model.smpl import SMPLServer
# from lib.model.smplx import SMPLServer
from lib.model.mesh import generate_mesh
from lib.model.sample import PointOnBones
from lib.model.generator_2d import Generator
from lib.model.network import ImplicitNetwork
from lib.model.superresolution import SuperresolutionHybrid4X
from lib.model.helpers import expand_cond, vis_images
from lib.utils.render import render_mesh_dict, weights2colors, render_point_dict
from lib.model.deformer import skinning
from lib.model.ray_tracing import DepthModule
from lib.utils.render import render_mesh_dict, Renderer
import time
# from lib.model.fast_deformer import ForwardDeformer

class BaseModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.opt = opt

        self.network = ImplicitNetwork(**opt.network)
        print(self.network)

        self.smpl_server = SMPLServer(gender='neutral')

        # self.deformer = ForwardDeformer(opt.deformer)
        self.deformer = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer)
        # assert False

        self.generator = Generator(opt.dim_shape)
        print(self.generator)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.part_points = [500, 500, 500, 500, 250]

        # self.offset_network = ImplicitNetwork(**opt.offset_network)

        # self.smpl_server = SMPLServer(gender='neutral')

        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        self.z_shapes = torch.nn.Embedding(meta_info.n_samples, opt.dim_shape)
        self.z_shapes.weight.data.fill_(0)

        self.z_details = torch.nn.Embedding(meta_info.n_samples, opt.dim_detail)
        self.z_details.weight.data.fill_(0)

        self.z_colors = torch.nn.Embedding(meta_info.n_samples, opt.dim_color)
        self.z_colors.weight.data.fill_(0)

        self.data_processor = data_processor

        if opt.stage=='fine':
            self.norm_network = ImplicitNetwork(**opt.norm_network)
            # print(self.norm_network)
            self.tex_network = ImplicitNetwork(**opt.tex_network)

            # self.superresolution = SuperresolutionHybrid4X(channels=32, img_channels=3)
            # print(self.superresolution)
            # self.raster_settings = PointsRasterizationSettings(image_size=1024,
            #     # radius=0.3 * (0.75 ** math.log2(600000 / 100)),
            #     radius=0.005,
            #     points_per_pixel=10
            # )
            # R = torch.from_numpy(np.array([[-1., 0., 0.],
            #                            [0., 1., 0.],
            #                            [0., 0., -1.]])).float().unsqueeze(0).cuda()
            # t = torch.from_numpy(np.array([[0., 0.3, 2.]])).float().cuda()
            # self.cameras = FoVOrthographicCameras(device='cuda', R=R, T=t)
            # self.compositor = AlphaCompositor(background_color=[1, 1, 1, 0]).cuda()
            # self.rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

            if opt.use_gan:
                from lib.model.losses import GANLoss
                self.gan_loss = GANLoss(self.opt)
                print(self.gan_loss.discriminator)

            if opt.use_gan_color:   # 给color的gan loss 
                from lib.model.losses import GANLoss
                self.gan_loss_color = GANLoss(self.opt)
                print(self.gan_loss_color.discriminator)
            
            if opt.use_perceptual:
                from lib.model.percep import VGGPerceptualLoss
                self.vgg_loss = VGGPerceptualLoss()

        self.render = DepthModule(**self.opt.ray_tracer)


    def configure_optimizers(self):

        grouped_parameters = self.parameters()
        
        def is_included(n): 
            if self.opt.stage =='fine':  # only train the z_details, z_colors, texture network, and normal network in the second stage
                if 'z_details' not in n and 'norm_network' not in n and 'z_colors' not in n and 'tex_network' not in n:
                    return False

            return True

        grouped_parameters = [
            {"params": [p for n, p in list(self.named_parameters()) if is_included(n)], 
            'lr': self.opt.optim.lr, 
            'betas':(0.9,0.999)},
        ]

        optimizer = torch.optim.Adam(grouped_parameters, lr=self.opt.optim.lr)

        if not self.opt.use_gan and not self.opt.use_gan_color:
            return optimizer
        elif self.opt.use_gan and not self.opt.use_gan_color:    # only use gan loss for normal
            optimizer_d = torch.optim.Adam(self.gan_loss.parameters(), 
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d
        elif not self.opt.use_gan and self.opt.use_gan_color:   # only use gan loss for color
            optimizer_d = torch.optim.Adam(self.gan_loss_color.parameters(), 
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d
        else:   # use gan loss for color and normal
            optimizer_d = torch.optim.Adam(itertools.chain(self.gan_loss.parameters(), self.gan_loss_color.parameters()),  
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d

    def forward(self, pts_d, smpl_tfs, smpl_verts, cond, canonical=False, canonical_shape=False, eval_mode=True, fine=False, mask=None, only_near_smpl=False, split=False, idx=None, return_pts=False):
        n_batch, n_points, n_dim = pts_d.shape

        outputs = {}        

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_d.device, dtype=torch.bool)

        # Filter based on SMPL
        if only_near_smpl:
            from kaolin.metrics.pointcloud import sided_distance
            distance, _ = sided_distance(pts_d, smpl_verts[:,::10])
            mask = mask & (distance<0.1*0.1)

        if not mask.any(): 
            return {'occ': -1000*torch.ones( (n_batch, n_points, 1), device=pts_d.device)}

        if canonical_shape:
            pts_c = pts_d 
            occ_pd, feat_pd = self.network( # geometry network
                                    pts_c, 
                                    cond={'latent': cond['latent']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)
        elif canonical:
            pts_c = self.deformer.query_cano(pts_d,  # Given canonical (with betas) point return its correspondence in the shape neutral space
                                            {'betas': cond['betas']}, 
                                            mask=mask)
            
            occ_pd, feat_pd = self.network(
                                    pts_c, 
                                    cond={'latent': cond['latent']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)

        else:
            cond_tmp = cond.copy()  # 不用copy()改变cond_tmp时cond会跟着变
            cond_tmp['latent'] = cond['lbs']
            
            pts_c, others = self.deformer.forward(pts_d,
                                        cond_tmp,
                                        smpl_tfs,
                                        mask=mask,
                                        eval_mode=eval_mode)

            occ_pd, feat_pd = self.network(
                                        pts_c.reshape((n_batch, -1, n_dim)), 
                                        cond={'latent': cond['latent']},
                                        mask=others['valid_ids'].reshape((n_batch, -1)),
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)

            occ_pd = occ_pd.reshape(n_batch, n_points, -1, 1)
            feat_pd = feat_pd.reshape(n_batch, n_points, -1, feat_pd.shape[-1])

            # mode=max in SNARF(mode=softmax or max)
            occ_pd, idx_c = occ_pd.max(dim=2)
            

            feat_pd = torch.gather(feat_pd, 2, idx_c.unsqueeze(-1).expand(-1, -1, 1, feat_pd.shape[-1])).squeeze(2)
            pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2)
            
            ## occupancy网络中用到的valid_mask    外面算loss也要用
            valid_mask = torch.gather(others['valid_ids'], 2, idx_c)   # (bs,2250,1)

            outputs['valid_mask'] = valid_mask


        outputs['occ'] = occ_pd
        outputs['pts_c'] = pts_c

        outputs['weights'] = self.deformer.query_weights(pts_c,  
                                                        cond={
                                                        'betas': cond['betas'],
                                                        'latent': cond['lbs']
                                                        })
        if fine:
            ## texture net
            # 输入geometry网络输出的特征、normal网络输出的特征、表情参数、z_color
            # 输出每个点的RGB值
            # condition = torch.cat([cond['color'], cond['thetas']], dim=-1)     # 64+56
            # condition = cond['color']
            condition = torch.cat([cond['color'], cond['lbs']], dim=-1)
            
            feature = feat_pd
            
            rgb_norm = self.tex_network(pts_c, 
                            cond={'latent': condition},
                            mask=mask,
                            input_feat=feature,
                            val_pad=1)

            outputs['color'] = rgb_norm[:, :, :3]
            outputs['norm'] = rgb_norm[:, :, 3:]

            # outputs['color'] = outputs_tex[:,:,:3]
            # outputs['density'] = outputs_tex[:,:,3:]
            # tex_features[mask] = tex_features[mask] / torch.linalg.norm(tex_features[mask],dim=-1,keepdim=True)
            # outputs['color'] = tex_features[:,:,:3]
            # if sr:
            #     # super-res module                                
            #     feature_map = tex_features.permute(0,2,1).reshape(-1,32,self.opt.img_res,self.opt.img_res)
            #     raw_image = feature_map[:,:3]
            #     image_sr = self.superresolution(raw_image, feature_map)
            #     image_min = image_sr.min()
            #     image_max = image_sr.max()
            #     outputs['image_sr'] = (2*image_sr - image_min - image_max) / (image_max - image_min)

            smpl_tfs = expand_cond(smpl_tfs, pts_c)[mask]
            # pts_homo = torch.ones_like(pts_c[:,:,0:1], device=pts_c.device)
            # outputs['deform_points'] = pts_c.clone()

            if not canonical:
                outputs['norm'][mask] = skinning(outputs['norm'][mask], outputs['weights'][mask], smpl_tfs, inverse=False, normal=True)
                # if flip != None:
                #     if flip[0]:
                #         outputs['norm'][mask][:,1:] *= -1
            outputs['norm'][mask] = outputs['norm'][mask] / torch.linalg.norm(outputs['norm'][mask],dim=-1,keepdim=True)
            # outputs['deform_points'][mask] = skinning(outputs['deform_points'][mask], outputs['weights'][mask], smpl_tfs, inverse=False)
            # points_new = Pointclouds(points=pts_c, normals=outputs['norm'], features=outputs['color'])
            # image_rgb = render_pytorch3d_point(points_new, mode='t', renderer_new=self.renderer)

            # outputs['color'][mask] = outputs['color'][mask] / torch.linalg.norm(outputs['color'][mask],dim=-1,keepdim=True)
        return outputs

    def point_render(self, points, radius=None):
        R = torch.from_numpy(np.array([[-1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., -1.]])).float().unsqueeze(0).to(points.device)
        t = torch.from_numpy(np.array([[0., 0.3, 2.]])).float().to(points.device)
        cameras = FoVOrthographicCameras(device=points.device, R=R, T=t)
        compositor = AlphaCompositor(background_color=[1, 1, 1, 0]).to(points.device)
        raster_settings = PointsRasterizationSettings(image_size=1024,
            # radius=0.3 * (0.75 ** math.log2(600000 / 100)),
            # radius=radius[0].permute(1, 0),
            radius=0.003,
            points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(points)
        r = raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        alphas = 1 - dists2 / (r * r)
        image_color, _ = compositor(
                    fragments.idx.long().permute(0, 3, 1, 2),
                    alphas,
                    points.features_packed().permute(1, 0),
                )
        return image_color
    # def forward_image_diff(self, v_pos, mesh_t_pos_idx_fx3, resolution=256, spp=1):
    #     # assert not hierarchical_mask
    #     self.ctx = dr.RasterizeGLContext(v_pos.device)
    #     assert False

        # mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        # v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        # v_pos_clip = self.camera.project(v_pos)  # Projection in the camera, gdna use othogonal cam

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        # num_layers = 1
        # mask_pyramid = None
        # assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        # # mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd, v_pos], dim=-1)  # Concatenate the pos  compute the supervision

        # with dr.DepthPeeler(self.ctx, v_pos, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
        #     for _ in range(num_layers):
        #         rast, db = peeler.rasterize_next_layer()
        #         # gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
        #         gb_feat, _ = interpolate(v_pos, rast, mesh_t_pos_idx_fx3)
        #         print(gb_feat.shape)
        #         assert False

        # hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        # antialias_mask = dr.antialias(
        #     hard_mask.clone().contiguous(), rast, v_pos,
        #     mesh_t_pos_idx_fx3)

        # depth = gb_feat[..., -2:-1]
        # ori_mesh_feature = gb_feat[..., :-4]
        # return ori_mesh_feature, antialias_mask, hard_mask, rast, mask_pyramid, depth

    def forward_2d(self, smpl_tfs, smpl_verts, cond, eval_mode=True, fine=True, res=256):

        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        pix_d = torch.stack([xv, yv], dim=-1).type_as(smpl_tfs)
        pix_d = pix_d.reshape(1,res*res,2)
        # ray_ori = torch.tensor([[ 0.0000, -0.3000,  2],]).unsqueeze(1).repeat(1, pix_d.shape[1], 1).type_as(smpl_tfs)

        def occ(x, mask=None):

            outputs = self.forward(x, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, mask=mask, fine=False, only_near_smpl=True)

            if mask is not None:
                return outputs['occ'][mask].reshape(-1, 1)
            else:
                return outputs['occ']        

        pix_d = torch.stack([pix_d[...,0], -pix_d[...,1]-0.3, torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        ray_dirs = torch.zeros_like(pix_d)
        ray_dirs[...,-1] = -1
        # ray_dirs = pix_d - ray_ori
        # ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        d = self.render(pix_d, ray_dirs, occ).detach()
        # pt_pred = pix_d + d.unsqueeze(-1) * ray_dirs
        
        pix_d[...,-1] += d*ray_dirs[...,-1]

        mask = ~d.isinf()
        outputs = self.forward(pix_d, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, fine=fine, mask=mask)

        outputs['mask'] = mask

        outputs['pts_c'][~mask, :] = 1

        img = outputs['pts_c'].reshape(res,res,3).data.cpu().numpy()
        # imageio.imsave('/mnt/sdb/zwt/gdna_addtex/projection.png', (255*img).astype(np.uint8)) 
        # assert False
        mask = outputs['mask'].reshape(res,res,1).data.cpu().numpy()

        img_mask = np.concatenate([img,mask],axis=-1)

        return img_mask

    def extract_points(self, smpl_tfs, smpl_verts, cond, radius=0.07, eval_mode=True, fine=True, num_points=771680, init_points=None, use_init=False):
        smpl_tfs_tmp = smpl_tfs.clone()
        smpl_verts_tmp = smpl_verts.clone()
        if use_init:
            init_verts = init_points
            # init_verts = smpl_verts.clone()
        else:
            init_verts = smpl_verts.clone()
        num_verts = init_verts.shape[1]
        if use_init:
            batch_size = 2
        else:
            batch_size = 8
        repeat_time = int((num_points//num_verts) / batch_size) * batch_size
        # repeat_time = 1
        pts_c = init_verts.expand(repeat_time, -1, -1).contiguous()
        pts_c += radius * torch.randn(pts_c.shape, device=init_verts.device)
        
        smpl_tfs_tmp = smpl_tfs_tmp.expand(batch_size, -1, -1, -1).contiguous()
        smpl_verts_tmp = smpl_verts_tmp.expand(batch_size, -1, -1).contiguous()
        cond_tmp = {}

        for con in cond:
            if con == 'latent':
                cond_tmp[con]= [cond[con][0].expand(batch_size, -1, -1, -1).contiguous(), cond[con][1].expand(batch_size, -1, -1, -1).contiguous()]
            else:
                cond_tmp[con] = cond[con].expand(batch_size, -1).contiguous()
            
        masks = []
        pointclouds = []
        weights = []
        for pts_c_split in torch.split(pts_c, batch_size, dim=0):
            outputs = self.forward(pts_c_split, smpl_tfs_tmp, smpl_verts_tmp, cond_tmp, eval_mode=eval_mode, fine=fine, mask=None, only_near_smpl=False, canonical=True)
            # outputs.append(output)
            masks.append(outputs['occ'])
            weights.append(outputs['weights'])
            pointclouds.append(outputs['pts_c'])
        # outputs['mask'] = mask

        # outputs['pts_c'][~mask, :] = 1
        # pointcloud = torch.stack(pointclouds, dim=0).reshape(1, -1, 3).data.cpu().numpy()
        # mask = torch.stack(masks, dim=0).reshape(1, -1, 1).data.cpu().numpy()
        pointcloud = torch.stack(pointclouds, dim=0).reshape(1, -1, 3)
        mask = torch.stack(masks, dim=0).reshape(1, -1, 1)
        weights = torch.stack(weights, dim=0).reshape(1, -1, 24)
        valid_mask = (mask>0)
        valid_points = pointcloud[valid_mask[:, :, 0]].unsqueeze(0)
        mask = mask[valid_mask[:, :, 0]].unsqueeze(0)
        weights = weights[valid_mask[:, :, 0]].unsqueeze(0)
        # valid_points = pointcloud[valid_mask[:, :, 0]].unsqueeze(0)
        # mask = mask[valid_mask[:, :, 0]].unsqueeze(0)
        # pts_c = pts_c.reshape(1, -1, 3)
        # pts_c = pts_c[valid_mask[:, :, 0]].unsqueeze(0)

        # points_mask = np.concatenate([valid_points,mask],axis=-1)
        # points_mask = torch.cat([valid_points,mask], -1)

        return valid_points, weights

    def extract_occ(self, smpl_tfs, smpl_verts, cond, radius=0.07, eval_mode=True, fine=True, num_points=771680, init_points=None, use_init=False):
        smpl_tfs_tmp = smpl_tfs.clone()
        smpl_verts_tmp = smpl_verts.clone()
        if use_init:
            init_verts = init_points
            # init_verts = smpl_verts.clone()
        else:
            init_verts = smpl_verts.clone()
        num_verts = init_verts.shape[1]
        if use_init:
            batch_size = 2
        else:
            batch_size = 8
        repeat_time = int((num_points//num_verts) / batch_size) * batch_size
        # repeat_time = 1
        pts_c = init_verts.expand(repeat_time, -1, -1).contiguous()
        pts_c += radius * torch.randn(pts_c.shape, device=init_verts.device)
        
        smpl_tfs_tmp = smpl_tfs_tmp.expand(batch_size, -1, -1, -1).contiguous()
        smpl_verts_tmp = smpl_verts_tmp.expand(batch_size, -1, -1).contiguous()
        cond_tmp = {}

        for con in cond:
            if con == 'latent':
                cond_tmp[con]= [cond[con][0].expand(batch_size, -1, -1, -1).contiguous(), cond[con][1].expand(batch_size, -1, -1, -1).contiguous()]
            else:
                cond_tmp[con] = cond[con].expand(batch_size, -1).contiguous()
            
        masks = []
        pointclouds = []
        weights = []
        for pts_c_split in torch.split(pts_c, batch_size, dim=0):
            outputs = self.forward(pts_c_split, smpl_tfs_tmp, smpl_verts_tmp, cond_tmp, eval_mode=eval_mode, fine=fine, mask=None, only_near_smpl=False, canonical=True)
            # outputs.append(output)
            masks.append(outputs['occ'])
            weights.append(outputs['weights'])
            pointclouds.append(outputs['pts_c'])
        # outputs['mask'] = mask

        # outputs['pts_c'][~mask, :] = 1
        # pointcloud = torch.stack(pointclouds, dim=0).reshape(1, -1, 3).data.cpu().numpy()
        # mask = torch.stack(masks, dim=0).reshape(1, -1, 1).data.cpu().numpy()
        pointcloud = torch.stack(pointclouds, dim=0).reshape(1, -1, 3)
        mask = torch.stack(masks, dim=0).reshape(1, -1, 1)
        weights = torch.stack(weights, dim=0).reshape(1, -1, 24)
        # if not use_init:
        valid_mask = (mask>0)
        valid_points = pointcloud[valid_mask[:, :, 0]].unsqueeze(0)
        mask = mask[valid_mask[:, :, 0]].unsqueeze(0)
        weights = weights[valid_mask[:, :, 0]].unsqueeze(0)
        # else:
            # valid_points = pointcloud
        return valid_points, weights, mask

    def forward_points(self, smpl_tfs, smpl_verts, cond, radius=0.04, eval_mode=True, fine=True, num_points=771680, init_points=None, use_init=False):

        # yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        # pix_d = torch.stack([xv, yv], dim=-1).type_as(smpl_tfs)
        # pix_d = pix_d.reshape(1,res*res,2)
        # ray_ori = torch.tensor([[ 0.0000, -0.3000,  2],]).unsqueeze(1).repeat(1, pix_d.shape[1], 1).type_as(smpl_tfs)

        smpl_tfs_tmp = smpl_tfs.clone()
        smpl_verts_tmp = smpl_verts.clone()
        if use_init:
            init_verts = init_points
            # init_verts = smpl_verts.clone()
        else:
            init_verts = smpl_verts.clone()
        num_verts = init_verts.shape[1]
        if use_init:
            batch_size = 2
        else:
            batch_size = 8
        repeat_time = int((num_points//num_verts) / batch_size) * batch_size
        # repeat_time = 1
        pts_d = init_verts.expand(repeat_time, -1, -1).contiguous()
        pts_d += radius * torch.randn(pts_d.shape, device=init_verts.device)
        
        # def occ(x, mask=None):

        #     outputs = self.forward(x, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, mask=mask, fine=False, only_near_smpl=True)

        #     if mask is not None:
        #         return outputs['occ'][mask].reshape(-1, 1)
        #     else:
        #         return outputs['occ']        

        # pix_d = torch.stack([pix_d[...,0], -pix_d[...,1]-0.3, torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        # ray_dirs = torch.zeros_like(pix_d)
        # ray_dirs[...,-1] = -1
        # ray_dirs = pix_d - ray_ori
        # ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        # d = self.render(pix_d, ray_dirs, occ).detach()
        # pt_pred = pix_d + d.unsqueeze(-1) * ray_dirs
        
        # pix_d[...,-1] += d*ray_dirs[...,-1]

        # mask = ~d.isinf()
        # outputs = []
        smpl_tfs_tmp = smpl_tfs_tmp.expand(batch_size, -1, -1, -1).contiguous()
        smpl_verts_tmp = smpl_verts_tmp.expand(batch_size, -1, -1).contiguous()
        cond_tmp = {}
        # cond_tmp['latent'][0] = cond['latent'][0][:1]
        # cond_tmp['latent'][1] = cond['latent'][1][:1]

        for con in cond:
            if con == 'latent':
                cond_tmp[con]= [cond[con][0].expand(batch_size, -1, -1, -1).contiguous(), cond[con][1].expand(batch_size, -1, -1, -1).contiguous()]
                # cond_tmp[con][1] = cond[con][1].expand(batch_size, -1, -1, -1).contiguous()
            else:
                # print(cond_tmp[con].shape)
                cond_tmp[con] = cond[con].expand(batch_size, -1).contiguous()
            # cond[con] = cond[con].expand(16, -1)
        masks = []
        pointclouds = []
        for pts_d_split in torch.split(pts_d, batch_size, dim=0):
            # print(pts_d_split.shape)
            outputs = self.forward(pts_d_split, smpl_tfs_tmp, smpl_verts_tmp, cond_tmp, eval_mode=eval_mode, fine=fine, mask=None, only_near_smpl=False)
            # outputs.append(output)
            masks.append(outputs['occ'])
            pointclouds.append(outputs['pts_c'])
        # outputs['mask'] = mask

        # outputs['pts_c'][~mask, :] = 1
        # pointcloud = torch.stack(pointclouds, dim=0).reshape(1, -1, 3).data.cpu().numpy()
        # mask = torch.stack(masks, dim=0).reshape(1, -1, 1).data.cpu().numpy()
        pointcloud = torch.stack(pointclouds, dim=0).reshape(1, -1, 3)
        mask = torch.stack(masks, dim=0).reshape(1, -1, 1)
        valid_mask = (mask>0)
        valid_points = pointcloud[valid_mask[:, :, 0]].unsqueeze(0).data.cpu().numpy()
        mask = mask[valid_mask[:, :, 0]].unsqueeze(0).data.cpu().numpy()
        # valid_points = pointcloud[valid_mask[:, :, 0]].unsqueeze(0)
        # mask = mask[valid_mask[:, :, 0]].unsqueeze(0)
        pts_d = pts_d.reshape(1, -1, 3)
        pts_d_init = pts_d[valid_mask[:, :, 0]].unsqueeze(0)

        points_mask = np.concatenate([valid_points,mask],axis=-1)
        # points_mask = torch.cat([valid_points,mask], -1)

        return points_mask, pts_d_init

    def forward_volume_rendering(self, smpl_tfs, smpl_verts, cond, eval_mode=True, fine=True, res=256, depth_res=8):

        batch_size = smpl_tfs.shape[0]
        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        pix_d = torch.stack([xv, yv], dim=-1).type_as(smpl_tfs)
        pix_d = pix_d.reshape(1,res*res,2).expand(batch_size, -1, -1)
        # ray_ori = torch.tensor([[ 0.0000, -0.3000,  2],]).unsqueeze(1).repeat(1, pix_d.shape[1], 1).type_as(smpl_tfs)
        def occ(x, mask=None):

            outputs = self.forward(x, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, mask=mask, fine=False, only_near_smpl=True)

            if mask is not None:
                return outputs['occ'][mask].reshape(-1, 1)
            else:
                return outputs['occ']        

        pix_d = torch.stack([pix_d[...,0], -pix_d[...,1]-0.3, torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        ray_dirs = torch.zeros_like(pix_d)
        ray_dirs[...,-1] = -1
        # ray_dirs = pix_d - ray_ori
        # ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)
        d = self.render(pix_d, ray_dirs, occ).detach()

        # mask = ~d.isinf()

        depths_coarse = torch.linspace(-0.01, 0.01, depth_res, device=ray_dirs.device)
        depth_delta = (0.02)/(depth_res - 1)
        depths_coarse += torch.rand_like(depths_coarse) * depth_delta
        depths_coarse = depths_coarse.view(1, 1, depth_res)
        d = d.unsqueeze(-1) + depths_coarse
        mask = ~d.isinf()
        # pix_z = pix_d[...,[-1]] + d*ray_dirs[...,[-1]]
        # assert False
        sigma = torch.sigmoid(depths_coarse / 0.003) / 0.003
        deltas = d[:, :, 1:] - d[:, :, :-1]
        density_delta = sigma * deltas
        alpha = 1 - torch.exp(-density_delta)
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        pix_d = pix_d.unsqueeze(2).expand(-1, -1, depth_res, -1)
        pix_d[..., [-1]] += (d*ray_dirs[...,[-1]]).unsqueeze(-1)
        depths = pix_d[..., [-1]]
        pix_d = pix_d.reshape(batch_size, -1, 3)
        # pix_d = pix_d.unsqueeze(-2) + depths_coarse*ray_dirs.unsqueeze(-2)

        # mask = ~d.isinf()
        outputs = self.forward(pix_d, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, fine=fine, mask=mask.reshape(batch_size,-1))
        outputs['mask'] = mask
        # colors = outputs['color']
        # weights = outputs['density']
        # norm = outputs['norm']
        # outputs['color'][~mask, :] = 1
        # outputs['density'][~mask, :] = -1000

        # deltas = depths[:, 1:, :] - depths[:, :-1, :]
        # colors_mid = (colors[:, :-1, :] + colors[:, 1:, :]) / 2
        # densities_mid = (densities[:, :-1, :] + densities[:, 1:, :]) / 2
        # depths_mid = (depths[:, :-1, :] + depths[:, 1:, :]) / 2

        # densities_mid = F.softplus(densities_mid - 1)
        # density_delta = densities_mid * deltas

        # alpha = 1 - torch.exp(-density_delta)

        # alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        # weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        # print(weights.shape)
        # print(colors_mid.shape)
        composite_rgb = (weights * colors).reshape(batch_size, -1, depth_res, 3)
        outputs['color'] = torch.sum(composite_rgb, -2)
        composite_norm = (weights * norm).reshape(batch_size, -1, depth_res, 3)
        outputs['norm'] = torch.sum(composite_norm, -2)
        # img = outputs['pts_c'].reshape(res,res,3).data.cpu().numpy()
        # imageio.imsave('/mnt/sdb/zwt/gdna_addtex/projection.png', (255*img).astype(np.uint8)) 
        # assert False
        # mask = outputs['mask'].reshape(res,res,1).data.cpu().numpy()

        # img_mask = np.concatenate([img,mask],axis=-1)

        return outputs

    def volume_render(self, smpl_tfs, smpl_verts, cond, eval_mode=True, fine=True, res=256, depth_res=2):

        batch_size = smpl_tfs.shape[0]
        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        pix_d = torch.stack([xv, yv], dim=-1).type_as(smpl_tfs)
        pix_d = pix_d.reshape(1,res*res,2).expand(batch_size, -1, -1)
        # ray_ori = torch.tensor([[ 0.0000, -0.3000,  2],]).unsqueeze(1).repeat(1, pix_d.shape[1], 1).type_as(smpl_tfs)
        def occ(x, mask=None):

            outputs = self.forward(x, smpl_tfs, smpl_verts, cond, eval_mode=True, mask=mask, fine=False, only_near_smpl=True)

            if mask is not None:
                return outputs['occ'][mask].reshape(-1, 1)
            else:
                return outputs['occ']        

        pix_d = torch.stack([pix_d[...,0], -pix_d[...,1]-0.3, torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        ray_dirs = torch.zeros_like(pix_d)
        ray_dirs[...,-1] = -1
        # ray_dirs = pix_d - ray_ori
        # ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)
        d = self.render(pix_d, ray_dirs, occ).detach()

        # mask = ~d.isinf()

        depths_coarse = torch.linspace(-0.02, 0.02, depth_res, device=ray_dirs.device)
        depth_delta = (0.04)/(depth_res - 1)
        depths_coarse += torch.rand_like(depths_coarse) * depth_delta
        depths_coarse = depths_coarse.view(1, 1, depth_res)
        d = d.unsqueeze(-1) + depths_coarse
        mask = ~d.isinf()
        # pix_z = pix_d[...,[-1]] + d*ray_dirs[...,[-1]]
        # assert False
        pix_d = pix_d.unsqueeze(2).expand(-1, -1, depth_res, -1)
        pix_d[..., [-1]] += (d*ray_dirs[...,[-1]]).unsqueeze(-1)
        depths = pix_d[..., [-1]]
        pix_d = pix_d.reshape(batch_size, -1, 3)
        # pix_d = pix_d.unsqueeze(-2) + depths_coarse*ray_dirs.unsqueeze(-2)

        # mask = ~d.isinf()
        outputs = self.forward(pix_d, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, fine=fine, mask=mask.reshape(batch_size,-1))
        outputs['mask'] = mask
        colors = outputs['color']
        weights = outputs['density']
        norm = outputs['norm']
        # outputs['color'][~mask, :] = 1
        # outputs['density'][~mask, :] = -1000

        # deltas = depths[:, 1:, :] - depths[:, :-1, :]
        # colors_mid = (colors[:, :-1, :] + colors[:, 1:, :]) / 2
        # densities_mid = (densities[:, :-1, :] + densities[:, 1:, :]) / 2
        # depths_mid = (depths[:, :-1, :] + depths[:, 1:, :]) / 2

        # densities_mid = F.softplus(densities_mid - 1)
        # density_delta = densities_mid * deltas

        # alpha = 1 - torch.exp(-density_delta)

        # alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        # weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        # print(weights.shape)
        # print(colors_mid.shape)
        composite_rgb = (weights * colors).reshape(batch_size, -1, depth_res, 3)
        outputs['color'] = torch.sum(composite_rgb, -2)
        composite_norm = (weights * norm).reshape(batch_size, -1, depth_res, 3)
        outputs['norm'] = torch.sum(composite_norm, -2)
        # img = outputs['pts_c'].reshape(res,res,3).data.cpu().numpy()
        # imageio.imsave('/mnt/sdb/zwt/gdna_addtex/projection.png', (255*img).astype(np.uint8)) 
        # assert False
        # mask = outputs['mask'].reshape(res,res,1).data.cpu().numpy()

        # img_mask = np.concatenate([img,mask],axis=-1)

        return outputs

    def prepare_cond(self, batch, train=True):

        cond = {}
        # cond['thetas'] =  batch['smpl_params'][:,7:-10]/np.pi
        cond['thetas'] =  batch['smpl_thetas'][:, 3:]/np.pi # for smplx
        # cond['betas'] = batch['smpl_params'][:,-10:]/10.
        cond['betas'] = batch['smpl_betas']/10. # for smplx
        # cond['expression'] = batch['smpl_expression']/10. # for smplx
        # cond['canon_thetas'] = batch['canon_thetas'][:, 3:]
        # if train:
        #     cond['smpl_betas_train'] = batch['smpl_betas_train']/10.

        z_shape = batch['z_shape']      
        cond['latent'] = self.generator(z_shape)
        cond['lbs'] = z_shape
        cond['detail'] = batch['z_detail']
        cond['color'] = batch['z_color']

        return cond
    

    def training_step_coarse(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0

        reg_shape = F.mse_loss(batch['z_shape'], torch.zeros_like(batch['z_shape']))
        self.log('reg_shape', reg_shape)
        loss = loss + self.opt.lambda_reg * reg_shape
        
        reg_lbs = F.mse_loss(cond['lbs'], torch.zeros_like(cond['lbs']))
        self.log('reg_lbs', reg_lbs)
        loss = loss + self.opt.lambda_reg * reg_lbs

        outputs = self.forward(batch['pts_d'], batch['smpl_tfs'], batch['smpl_verts'], cond, eval_mode=False, only_near_smpl=False)
        loss_bce = F.binary_cross_entropy_with_logits(outputs['occ'], batch['occ_gt'], pos_weight=batch['pos_weight'].mean())
        self.log('train_bce', loss_bce)
        loss = loss + loss_bce

        # Bootstrapping
        num_batch = batch['pts_d'].shape[0]

        # if self.deformer.opt.skinning_mode=='voxel':
        #     loss_tv = self.opt.lambda_tv*tv_loss(self.deformer.lbs_voxel,l2=True)*((self.deformer.opt.res//32)**3)
        #     self.log('loss_tv', loss_tv)
        #     loss = loss + loss_tv
        
        if self.current_epoch < self.opt.nepochs_pretrain:

            # Bone occupancy loss
            if self.opt.lambda_bone_occ > 0:
                pts_c, _, occ_gt, _ = self.sampler_bone.get_points(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                outputs = self.forward(pts_c, None, None, cond, canonical=True, only_near_smpl=False)
                loss_bone_occ = F.binary_cross_entropy_with_logits(outputs['occ'], occ_gt.unsqueeze(-1))
                self.log('train_bone_occ', loss_bone_occ)
                loss = loss + self.opt.lambda_bone_occ * loss_bone_occ

            # Joint weight loss
            if self.opt.lambda_bone_w > 0:
                pts_c, w_gt, _ = self.sampler_bone.get_joints(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                w_pd = self.deformer.query_weights(pts_c, {'latent': cond['lbs'], 'betas': cond['betas']*0})
                loss_bone_w = F.mse_loss(w_pd, w_gt)
                self.log('train_bone_w', loss_bone_w)
                loss = loss + self.opt.lambda_bone_w * loss_bone_w
            
        #     # lbs loss
        if self.opt.lambda_lbs > 0:
            w_smpl = self.deformer.query_weights(self.smpl_server.verts_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1), {'latent': cond['lbs'], 'betas': cond['betas']*0})
            w_smpl_gt = self.smpl_server.weights_c.type_as(batch['pts_d']).expand(num_batch, -1, -1)
            loss_lbs = F.mse_loss(w_smpl, w_smpl_gt)
            self.log('train_lbs_weight', loss_lbs)
            loss = loss + self.opt.lambda_lbs * loss_lbs

        # offsets_1 = self.offset_network(batch['smpl_verts_cano'], cond={'latent': cond['lbs'], 'pose':cond['canon_thetas']}, normalize=True, val_pad=0)
        # offsets = self.offset_network(batch['pts_d'], cond={'latent': cond['lbs'], 'pose':cond['thetas']}, normalize=True, val_pad=0)
        # reg_offsets = F.mse_loss(offsets, torch.zeros_like(offsets), reduction='sum')
        # self.log('reg_offset', reg_offsets)
        # loss = loss + self.opt.lambda_offset * reg_offsets

        # Displacement loss(not in snarf only in gdna, to train the beta)
        pts_c_gt = self.smpl_server.verts_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1)
        # if self.current_epoch < self.opt.nepochs_pretrain:
        # pts_c = self.deformer.query_cano(batch['smpl_verts_cano'], {'betas': cond['betas']})
            # pts_c = self.deformer.query_cano(batch['smpl_verts_beta'], {'betas': cond['smpl_betas_train']})
        # else:
        pts_c = self.deformer.query_cano(batch['smpl_verts_cano'], {'betas': cond['betas']})
        loss_disp = F.mse_loss(pts_c, pts_c_gt)

        self.log('train_disp', loss_disp)
        loss = loss + self.opt.lambda_disp * loss_disp

        return loss

    def training_step_fine(self, batch, batch_idx, optimizer_idx=None, batch_points=40000):
        
        cond = self.prepare_cond(batch)
        renderer = Renderer(image_size=1024, real_cam=False, point=True, device=batch['smpl_tfs_img'].device)

        loss = 0

        if self.opt.volume_rendering:
            outputs = self.volume_render(batch['smpl_tfs_img'], batch['smpl_verts'], cond, eval_mode=True, fine=True, res=self.opt.img_res)
        else:
            norm = []
            rgb = []
            for pts_c_split in torch.split(batch['cache_pts'], batch_points, dim=1):
                output = self.forward(pts_c_split, batch['smpl_tfs_img'], None, cond, canonical_shape=True, mask=None, fine=True, return_pts=True)
                norm.append(output['norm'])
                rgb.append(output['color'])
        
        normals = torch.cat(norm, dim=1)
        colors = torch.cat(rgb, dim=1)
        alphas = torch.ones_like(colors[:,:,0:1])
        feats = torch.cat([colors, normals, alphas], dim=-1)
        points_feats = Pointclouds(points=batch['cache_ptsd'], normals=normals, features=feats)
        # mask = points_colors[:, :, 3:4].unsqueeze(0).permute(0, 3, 1, 2)
        # points_normals = Pointclouds(points=batch['cache_ptsd'], normals=normals, features=normals)
        # images_norm = self.point_render(points_normals, batch['cache_radius'])
        # images_rgb = self.point_render(points_colors, batch['cache_radius'])
        images = render_point_dict(points_feats, renderer)
        # images = self.point_render(points_feats)
        # images_norm = self.point_render(points_normals)
        # images_rgb = self.point_render(points_colors)
        # images_rgb = render_pytorch3d_point(points_colors, mode='t', renderer_new=self.renderer, train=True)
        # images_mask = images_rgb[:,3:,:].expand(-1, 3, -1, -1)
        # images_rgb = images_rgb[:,:3,]
        images_rgb = images[:,:3]
        images_norm = images[:,3:6]
        images_mask = images[:,6:].expand(-1, 3, -1, -1)
        # images_norm = render_pytorch3d_point(points_normals, mode='t', renderer_new=self.renderer, train=True)
        
        self.gan_loss_input = {
            'real': batch['norm_img'],
            'fake': images_norm,
            # 'hf_mask': batch['hfmask_img']
        }

        self.gan_loss_input_color = {
            'real': batch['color_img'],
            # 'fake': outputs['image_sr'],
            'fake': images_rgb,
            # 'hf_mask': batch['hfmask_img']
        }

        ## photo loss
        mask_color = ~(self.gan_loss_input_color['real']==1)
        mask_color = mask_color * images_mask
        photo_loss_color = torch.nn.functional.mse_loss(self.gan_loss_input_color['fake']*mask_color, self.gan_loss_input_color['real']*mask_color)
        # photo_loss_color = torch.nn.functional.mse_loss(self.gan_loss_input_color['fake']*mask, self.gan_loss_input_color['real']*mask)
        self.log('loss_train/photo_loss_color', photo_loss_color)
        loss += photo_loss_color

        mask = ~(self.gan_loss_input['real']==1)
        mask = mask * images_mask
        photo_loss = torch.nn.functional.mse_loss(self.gan_loss_input['fake']*mask, self.gan_loss_input['real']*mask)
        self.log('loss_train/photo_loss', photo_loss) 
        loss += photo_loss

        if batch_idx%300 == 0 and self.trainer.is_global_zero:
            # normal
            img = vis_images(self.gan_loss_input)
            # self.logger.experiment.log({"imgs":[wandb.Image(img)]})                  
            save_path = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), (255*img).astype(np.uint8)) 
            # color
            gan_loss_input_color_vis = self.gan_loss_input_color.copy()
            gan_loss_input_color_vis['real'] = gan_loss_input_color_vis['real'] * mask_color
            gan_loss_input_color_vis['fake'] = gan_loss_input_color_vis['fake'] * mask_color

            img_rgb = vis_images(gan_loss_input_color_vis)     # 可视化时给rgb图片乘上mask（给normal图片不乘），对比看看
            # img_rgb = vis_images(self.gan_loss_input_color)
            # self.logger.experiment.log({"imgs_rgb":[wandb.Image(img_rgb)]})                  
            imageio.imsave(os.path.join(save_path,'%04d_rgb.png'%self.current_epoch), (255*img_rgb).astype(np.uint8)) 
        ## gan loss on rendered images
        # normal
        if self.opt.use_gan:
            loss_gan, log_dict = self.gan_loss(self.gan_loss_input, self.global_step, optimizer_idx, for_color=False)
            for key, value in log_dict.items(): self.log(key, value)
            loss += self.opt.lambda_gan*loss_gan
        # color  
        if self.opt.use_gan_color:
            loss_gan_color, log_dict_color = self.gan_loss_color(self.gan_loss_input_color, self.global_step, optimizer_idx, for_color=True)
            for key, value in log_dict_color.items(): self.log(key, value)
            loss += self.opt.lambda_gan_color*loss_gan_color
        
        if self.opt.use_perceptual:
            loss_vgg_color = self.vgg_loss(self.gan_loss_input_color['real']*images_mask, self.gan_loss_input_color['fake']*images_mask)
            loss += self.opt.lambda_vgg/2 * loss_vgg_color
            loss_vgg = self.vgg_loss(self.gan_loss_input['real']*mask, self.gan_loss_input['fake']*mask)
            loss += self.opt.lambda_vgg/2 * loss_vgg


        if optimizer_idx == 0 or (not self.opt.use_gan and not self.opt.use_gan_color):

            ## predicted normal vs gt normal loss, predicted color vs gt color loss
            if self.opt.norm_loss_3d or self.opt.color_loss_3d:
                outputs = self.forward(batch['pts_surf'], batch['smpl_tfs'],  batch['smpl_verts'], cond, canonical=False, fine=True)

            if self.opt.norm_loss_3d:           
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_surf'])).mean() 
            else:
                loss_norm = 0
            # else:
                # loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_img'].permute(0,2,3,1).flatten(1,2)))[batch['cache_mask']].mean()    
            self.log('loss_train/train_norm', loss_norm)
            loss += loss_norm * 0

            if self.opt.color_loss_3d:
                loss_color = torch.sum((outputs['color']-batch['color_surf'])**2, dim=-1)   # (4,2000,3)->(4,2000)
                loss_color = torch.sum(loss_color) / (outputs['color'].shape[0]*outputs['color'].shape[1]) #->(1,)
            else:   # TODO: 还没写
                loss_color = 0
            self.log('loss_train/train_color', loss_color)
            loss += self.opt.lambda_color * loss_color * 0

            
            ## regularization term
            reg_detail = torch.nn.functional.mse_loss(batch['z_detail'], torch.zeros_like(batch['z_detail']))
            self.log('loss_train/reg_detail', reg_detail)
            loss += self.opt.lambda_reg * reg_detail

            reg_color = torch.nn.functional.mse_loss(batch['z_color'], torch.zeros_like(batch['z_color']))
            self.log('loss_train/reg_color', reg_color)
            loss += self.opt.lambda_reg_color * reg_color

        return loss

    def evel_pointrendering(self, points, cond, smpl_tfs, pts_d, batch_points=40000, stage='fine'):
        # cond = self.prepare_cond(batch)

        # loss = 0
        renderer = Renderer(image_size=512, real_cam=False, point=True, device=points.device)
        # with torch.no_grad():
        if stage == 'fine':
            norm = []
            rgb = []
            for pts_c_split in torch.split(points, batch_points, dim=1):
                output = self.forward(pts_c_split, smpl_tfs, None, cond, canonical_shape=True, mask=None, fine=True, return_pts=True)
                norm.append(output['norm'])
                rgb.append(output['color'])
            normals = torch.cat(norm, dim=1)
            colors = torch.cat(rgb, dim=1)
            alphas = torch.ones_like(colors[:,:,0:1], requires_grad=True)
            # colors = torch.cat([colors, alphas], dim=-1)
            feats = torch.cat([colors, normals, alphas], dim=-1)
            points_feats = Pointclouds(points=pts_d, normals=normals, features=feats)
            #  mask = points_colors[:, :, 3:4].unsqueeze(0).permute(0, 3, 1, 2)
            # points_colors = Pointclouds(points=pts_d, normals=normals, features=colors)
            # mask = points_colors[:, :, 3:4].unsqueeze(0).permute(0, 3, 1, 2)
            # points_normals = Pointclouds(points=pts_d, normals=normals, features=normals)
            # images_norm = self.point_render(points_normals, batch['cache_radius'])
            # images_rgb = self.point_render(points_colors, batch['cache_radius'])
            images = render_point_dict(points_feats, renderer)
            images_rgb = images[:,:3]
            images_norm = images[:,3:6]
            images_mask = images[:,6:].expand(-1, 3, -1, -1) 
        else:
            occ = []
            for pts_c_split in torch.split(points, batch_points, dim=1):
                output = self.forward(pts_c_split, smpl_tfs, None, cond, canonical_shape=True, mask=None, fine=False, return_pts=True)
                occ.append(output['occ'])
            occs = torch.cat(occ, dim=1)
            # occs[occs<0] += -100
            # colors = torch.cat([colors, alphas], dim=-1)
            # feats = torch.cat([occs], dim=-1)
            # occs = self.sigmoid(occs)
            occs = self.relu(occs).clamp(max=1.)
            points_feats = Pointclouds(points=pts_d, features=occs)
            images_mask = render_point_dict(points_feats, renderer, color=False)
            # images_mask = images_mask.expand(-1, 3, -1, -1) 
            images_norm = None
            images_rgb = None
            # images_norm = self.point_render(points_normals)
            # images_rgb = self.point_render(points_colors)
        # images_rgb = render_pytorch3d_point(points_colors, mode='t', renderer_new=self.renderer, train=True)
        # images_mask = images_rgb[:,3:,:].expand(-1, 3, -1, -1)
        # images_rgb = images_rgb[:,:3,]
        # images_norm = render_pytorch3d_point(points_normals, mode='t', renderer_new=self.renderer, train=True)

        return images_norm, images_rgb, images_mask

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server, load_volume=self.opt.stage!='fine')

        batch['z_shape'] = self.z_shapes(batch['index'])
        batch['z_detail'] = self.z_details(batch['index'])
        batch['z_color'] = self.z_colors(batch['index'])

        if not self.opt.stage=='fine':
            loss = self.training_step_coarse(batch, batch_idx)
        else:
            loss = self.training_step_fine(batch, batch_idx, optimizer_idx=optimizer_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):

        # Data prep
        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server)

        batch['z_shape'] = self.z_shapes(batch['index'])
        batch['z_detail'] = self.z_details(batch['index'])
        batch['z_color'] = self.z_colors(batch['index'])

        if batch_idx == 0 and self.trainer.is_global_zero:
            with torch.no_grad(): self.plot(batch)   

    def extract_mesh(self, smpl_verts, smpl_tfs, cond, res_up=3):
        def occ_func(pts_c):
            # print('occ_func')
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False)
            return outputs['occ'].reshape(-1,1)

        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up)
       
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}
        
        verts  = mesh['verts'].unsqueeze(0)
        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, fine=self.opt.stage=='fine', only_near_smpl=False)
        
        mesh['weights'] = outputs['weights'][0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        
        mesh['pts_c'] = outputs['pts_c'][0].detach()
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['color'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 
        return mesh

    def deform_mesh(self, mesh, smpl_tfs):
        import copy
        # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
        mesh = copy.deepcopy(mesh)
        smpl_tfs = smpl_tfs.expand(mesh['verts'].shape[0],-1,-1,-1)
        mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs)
        
        if 'norm' in mesh:
            mesh['norm']  = skinning(mesh['norm'], mesh['weights'], smpl_tfs, normal=True)
            mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
        return mesh
    
    def deform_point(self, points, weights, smpl_tfs):
        smpl_tfs = smpl_tfs.expand(points.shape[1],-1,-1,-1)
        points = skinning(points[0], weights[0], smpl_tfs)
            
        return points.unsqueeze(0)
    
    # def deform_mesh(self, points, smpl_tfs, weights, norm):
    #     import copy
    #     # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
    #     smpl_tfs = smpl_tfs.expand(points.shape[0],-1,-1,-1)
    #     mesh['verts'] = skinning(points, weights, smpl_tfs)
        
    #     if 'norm' in mesh:
    #         mesh['norm']  = skinning(mesh['norm'], mesh['weights'], smpl_tfs, normal=True)
    #         mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
    #     return mesh

    def deform_points(self, points, smpl_tfs, weights, norm):
        import copy
        # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
        smpl_tfs = smpl_tfs.expand(points.shape[0],-1,-1,-1)
        points_deform = skinning(points, weights, smpl_tfs)
        
        # if 'norm' in mesh:
        norm  = skinning(norm, weights, smpl_tfs, normal=True)
        norm = norm / torch.linalg.norm(norm,dim=-1,keepdim=True)
            
        return points_deform, norm

    def plot(self, batch):
        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                try:
                    batch[key] = batch[key][[0]]
                except:
                    print(key)
                    print(batch[key])
                    assert False

        cond = self.prepare_cond(batch)
        renderer = Renderer(image_size=512, real_cam=True, device=batch['smpl_tfs'].device)
        surf_pred_cano = self.extract_mesh(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3)
        surf_pred_def  = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'])

        img_list = []
        img_list.append(render_mesh_dict(surf_pred_cano,mode='npt',render_new=renderer))
        img_list.append(render_mesh_dict(surf_pred_def,mode='npt',render_new=renderer))

        img_all = np.concatenate(img_list, axis=1)

        # self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
        
        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), img_all) 

    def pca_sample(self, low_rank=32, batch_size=1, mode='g'):

        # idx = [6, 7, 103, 193]
        # idx = torch.tensor(idx).long()
        if mode == 'g':
            A = self.z_shapes.weight.clone().unsqueeze(1)
        else:
            A = self.z_colors.weight.clone().unsqueeze(1)
        num_subjects, num_vertices, dim = A.shape

        A = A.view(num_subjects, -1)

        (U, S, V) = torch.pca_lowrank(A, q=low_rank, center=True, niter=1)

        params = torch.matmul(A, V) # (N, 128)
        mean = params.mean(dim=0)
        cov = torch.cov(params.T)

        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        random_codes = m.sample((batch_size,)).to(A.device)

        return torch.matmul(random_codes.detach(), V.t()).view(-1, dim)

    def sample_codes(self, n_sample, std_scale=1):
        device = self.z_shapes.weight.device
        
        # mean_shapes = self.z_shapes.weight.data.mean(0)
        # std_shapes = self.z_shapes.weight.data.std(0)
        # mean_details = self.z_details.weight.data.mean(0)
        # std_details = self.z_details.weight.data.std(0)
        # mean_colors = self.z_colors.weight.data.mean(0)
        # std_colors = self.z_colors.weight.data.std(0)
        # idx = [6, 7, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63, 64, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 193, 194]
        # idx = [58, 30]
        # idx = torch.tensor(idx).long()
        mean_shapes = self.z_shapes.weight.data.mean(0)
        std_shapes = self.z_shapes.weight.data.std(0)
        mean_details = self.z_details.weight.data.mean(0)
        std_details = self.z_details.weight.data.std(0)
        mean_colors = self.z_colors.weight.data.mean(0)
        std_colors = self.z_colors.weight.data.std(0)

        z_shape = torch.randn(n_sample, self.opt.dim_shape, device=device)
        z_detail = torch.randn(n_sample, self.opt.dim_detail, device=device)  
        z_color = torch.randn(n_sample, self.opt.dim_color, device=device) 
        
        z_shape = z_shape*std_shapes*std_scale+mean_shapes
        z_detail = z_detail*std_details*std_scale+mean_details
        z_color = z_color*std_colors*std_scale+mean_colors

        return z_shape, z_detail, z_color

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)