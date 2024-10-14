import os
import cv2
import torch
import kaolin
import pandas
import imageio
import numpy as np
from tqdm import tqdm
import csv
import trimesh
from smplx import SMPL
# from lib.model.smpl import SMPLServer
import json

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io.obj_io import load_obj, load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points, knn_gather
from pytorch3d.renderer import TexturesUV, TexturesVertex
from trimesh.smoothing import filter_humphrey

import sys
sys.path.append('.')

from lib.utils.render import render_pytorch3d, Renderer, render_pytorch3d_point
from lib.utils.uv import Uv2Attr

class ScanProcessor():

    def __init__(self):

        self.scan_folder =  '/mnt/sdb/zwt/dataset/ThumanDataset/THuman2.0_Release/'

        self.smpl_folder =  './data/THuman2.0_smpl'

        self.scan_list = sorted(os.listdir(self.scan_folder))

        self.output_folder = './data/THuman2.0_processed_debug'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.renderer = Renderer(1024)
        self.smpl = SMPL('/mnt/sdb/zwt/PARE/data/body_models/smpl', create_transl=False)
        # self.smpl_faces = self.smpl.faces
        # self.smpl_weights = self.smpl.lbs_weights.cuda()

        smpl_seg = json.load(open('/mnt/sdb/zwt/gdna_addtex/smpl_vert_segmentation.json'))
        self.hand_verts = smpl_seg['rightHand'] + smpl_seg['leftHand'] + smpl_seg['rightHandIndex1'] + smpl_seg['leftHandIndex1']
        # self.head_verts = smpl_seg['head']
        smpl_output_canon = self.smpl(betas=torch.zeros((1,10)).float(), body_pose=torch.zeros((1,69)).float(), global_orient=torch.zeros((1,3)).float())
        smpl_verts_canon = smpl_output_canon.vertices
        self.face_verts_idx = [i for i in smpl_seg['head'] if smpl_verts_canon[0][i][2] > -0.03]
        self.hair_verts_idx = [i for i in smpl_seg['head'] if smpl_verts_canon[0][i][2] < -0.03]

    def process(self, index, mirror=False):

        batch = {}

        scan_name = "%04d"%index

        scan_path = os.path.join(self.scan_folder,scan_name, scan_name+'.obj')

        if mirror:
            output_folder = os.path.join(self.output_folder, scan_name+'_mir')
        else:
            output_folder = os.path.join(self.output_folder, scan_name)
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        batch['scan_name'] = scan_name

        pickle_path = os.path.join(self.smpl_folder, '%04d_smpl.pkl'%index)
        file = pandas.read_pickle(pickle_path)
        smpl_param = np.concatenate([np.ones( (1,1)), # scale
                                np.zeros( (1,3)),   # transl
                                file['global_orient'].reshape(1,-1),
                                file['body_pose'].reshape(1,-1),
                                file['betas'][:,:10]], axis=1)[0]

        if mirror:
            flipped_smpl_param = smpl_param.copy()
            flipped_smpl_param[4:76] = flip_smpl_pose(flipped_smpl_param[4:76])
            batch['smpl_params'] = flipped_smpl_param 
        else:
            batch['smpl_params'] = smpl_param

        # param_canonical = torch.zeros((1, 86),dtype=torch.float32)
        # param_canonical[0, 9] = np.pi / 6
        # param_canonical[0, 12] = -np.pi / 6
        # smpl_verts_canon = self.smpl(betas=torch.tensor(file['betas'][:,:10]).float(), body_pose=param_canonical[:,7:76].float(), global_orient=param_canonical[:,4:7].float()).vertices
        # smpl_verts_canon = self.smpl(betas=torch.tensor(file['betas'][:,:10]).float()*0, body_pose=param_canonical[:,7:76].float(), global_orient=param_canonical[:,4:7].float()).vertices
        # print(0.5 * (smpl_verts_canon[:,:,0].min()+smpl_verts_canon[:,:,0].max()))
        # print(0.5 * (smpl_verts_canon[:,:,1].min()+smpl_verts_canon[:,:,1].max()))
        # print(0.5 * (smpl_verts_canon[:,:,2].min()+smpl_verts_canon[:,:,2].max()))
        # assert False
        smpl_verts = self.smpl(betas=torch.tensor(file['betas'][:,:10]).float(), body_pose=torch.tensor(file['body_pose'].reshape(1,-1)).float(), global_orient=torch.tensor(file['global_orient'].reshape(1,-1)).float()).vertices
        # print(smpl_verts[:,:,0].min())
        # print(smpl_verts[:,:,0].max())
        # print(smpl_verts[:,:,1].min())
        # print(smpl_verts[:,:,1].max())
        # print(smpl_verts[:,:,2].min())
        # print(smpl_verts[:,:,2].max())
        # smpl_verts_mid = torch.tensor([[0.5*(smpl_verts[:,:, 0].min()+smpl_verts[:,:, 0].max()), 0.5*(smpl_verts[:,:, 1].min()+smpl_verts[:,:, 1].max()), 0.5*(smpl_verts[:,:, 2].min()+smpl_verts[:,:, 2].max())],])
        # smpl_hf_verts = torch.cat([smpl_verts[:, self.hand_verts], smpl_verts[:, self.head_verts]], dim=1).cuda() # head and face verts
        smpl_verts_label = torch.zeros_like(smpl_verts)[:,:,0:1].cuda()
        smpl_verts_label[:, self.hand_verts] = 1
        smpl_verts_label[:, self.face_verts_idx] = 2
        smpl_verts_label[:, self.hair_verts_idx] = 3

        scan_verts_ori, faces, aux = load_obj(scan_path, 
                                                device=torch.device("cuda:0"),
                                                load_textures=True)

        # mesh = load_objs_as_meshes([scan_path], device=torch.device("cuda:0"))
        # scan_verts = mesh.verts_list()[0]
        # scan_faces = mesh.faces_list()[0]
        # texture = mesh.textures

        scan_faces_ori = faces.verts_idx.long()

        ## 读取mesh每个点的color值
        scan_uvs = aux.verts_uvs     # mesh上每个顶点的uv坐标
        uv_img = aux.texture_images['material0'].to(scan_uvs.device)     # 保存texture的uv图
        uv_img = torch.flip(uv_img, (0,))   # 为了适应Uv2Attr，需上下翻转
        uv_size = uv_img.shape[0]
        uv_reader = Uv2Attr(torch.round(scan_uvs.unsqueeze(0) * uv_size), size=uv_size)
        scan_colors = uv_reader(uv_img.unsqueeze(0).permute(0,3,1,2), bilinear=True).permute(0, 2, 1).contiguous()   # 值在[0,1]范围
        # scan_verts_ori = scan_verts_ori - torch.tensor(file['transl']).cuda().float().expand(scan_verts_ori.shape[0], -1)
        # scan_verts_ori = scan_verts_ori/file['scale'][0]
        # print(scan_verts_ori.shape)
        # print(torch.tensor([[0.5*(scan_verts_ori[:,0].min()+scan_verts_ori[:,0].max()), 0.5*(scan_verts_ori[:,1].min()+scan_verts_ori[:,1].max()),0.5*(scan_verts_ori[:,2].min()+scan_verts_ori[:,2].max())],]).float())
        # assert False
        meshexport = trimesh.Trimesh(scan_verts_ori.cpu(), scan_faces_ori.cpu())
        connected_comp = meshexport.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        meshexport_verts = torch.tensor(max_comp.vertices)
        scan_faces = torch.tensor(max_comp.faces).cuda()
        # print(meshexport_verts.shape)
        # scan_colors = torch.tensor(max_comp.visual.vertex_colors[:,:3]/255).float().cuda()
        bboxes = torch.tensor([[0.5*(meshexport_verts[:,0].min()+meshexport_verts[:,0].max()), 0.5*(meshexport_verts[:,1].min()+meshexport_verts[:,1].max()),0.5*(meshexport_verts[:,2].min()+meshexport_verts[:,2].max())],]).float()
        # print(bboxes)
        # assert False
        # scale_x = (meshexport_verts[:,0].max() - meshexport_verts[:,0].min()) / (smpl_verts[:,:,0].max()-smpl_verts[:,:,0].min())
        # scale_y = (meshexport_verts[:,1].max() - meshexport_verts[:,1].min()) / (smpl_verts[:,:,1].max()-smpl_verts[:,:,1].min())
        # scale_z = (meshexport_verts[:,2].max() - meshexport_verts[:,2].min()) / (smpl_verts[:,:,2].max()-smpl_verts[:,:,2].min())
        # scale = (scale_x + scale_y + scale_z) / 3
        scale = torch.tensor(file['scale']).float().cuda()
        smpl_verts_scale = smpl_verts.cuda() * scale
        smpl_verts_mid = torch.tensor([[0.5*(smpl_verts_scale[:,:, 0].min()+smpl_verts_scale[:,:, 0].max()), 0.5*(smpl_verts_scale[:,:, 1].min()+smpl_verts_scale[:,:, 1].max()), 0.5*(smpl_verts_scale[:,:, 2].min()+smpl_verts_scale[:,:, 2].max())],])
        transl = bboxes - smpl_verts_mid
        # print(transl)
        # print(file['transl'])
        # assert False
        meshexport_verts = (meshexport_verts - transl) / scale.cpu()
        scan_verts = meshexport_verts.float().cuda()
        scan_verts_ori = (scan_verts_ori - transl.cuda()) / scale

        if mirror:
            scan_verts_flipped = scan_verts_ori.clone()
            scan_verts_flipped[:,0] *= -1
            # scan_verts_ori[:,0] *= -1
            # scan_faces_flipped = scan_faces.clone()
            # scan_faces_flipped = torch.flip(scan_faces_flipped, dims=(1,))
            batch['scan_verts'] = scan_verts_flipped.data.cpu().numpy()
        else:
            batch['scan_verts'] = scan_verts.data.cpu().numpy()
        batch['scan_faces'] = scan_faces.data.cpu().numpy()

        num_verts, num_dim = scan_verts.shape
        pts_ret = knn_points(scan_verts.unsqueeze(0), smpl_verts.cuda(), K=1, return_sorted=True)
        pts_label = knn_gather(smpl_verts_label, pts_ret.idx[:,:,0:1])[0,:,0,:]
        # pts_weight = knn_gather(self.smpl_weights.unsqueeze(0), pts_ret.idx)[0,:,:,:].reshape(num_verts, -1)
        num_hands = (pts_label==1).sum()
        num_heads = (pts_label==2).sum()
        num_body = (pts_label==0).sum()
        random_idx_body = torch.randint(0, num_body, [100000, 1], device=scan_verts.device)
        random_idx_hands = torch.randint(0, num_hands, [10000, 1], device=scan_verts.device)
        random_idx_head = torch.randint(0, num_heads, [20000, 1], device=scan_verts.device)
        
        pts_body = torch.gather(scan_verts[pts_label[:,0]==0], 0, random_idx_body.expand(-1, num_dim))
        pts_hands = torch.gather(scan_verts[pts_label[:,0]==1], 0, random_idx_hands.expand(-1, num_dim))
        pts_head = torch.gather(scan_verts[pts_label[:,0]==2], 0, random_idx_head.expand(-1, num_dim))
        # pts_body_w = torch.gather(pts_weight[pts_label[:,0]==0], 0, random_idx_body.expand(-1, num_dim*24))
        # pts_hands_w = torch.gather(pts_weight[pts_label[:,0]==1], 0, random_idx_hands.expand(-1, num_dim*24))
        # pts_head_w = torch.gather(pts_weight[pts_label[:,0]==2], 0, random_idx_head.expand(-1, num_dim*24))
        pts_surf = torch.cat([pts_body, pts_hands, pts_head], dim=0)
        # pts_surf_w = torch.cat([pts_body_w, pts_hands_w, pts_head_w], dim=0)
        
        pts_surf += 0.01 * torch.randn(pts_surf.shape, device=scan_verts.device)
        pts_bbox = torch.rand(pts_surf.shape, device=scan_verts.device) * 2 - 1
        # pts_ret_bb = knn_points((pts_bbox+bboxes).unsqueeze(0), smpl_verts.cuda(), K=3, return_sorted=True)
        # pts_weight_bb = knn_gather(self.smpl_weights.unsqueeze(0), pts_ret_bb.idx)[0,:,:,:].reshape(pts_surf.shape[0], -1)
        pts_d = torch.cat([pts_surf, pts_bbox],dim=0)
        # pts_w = torch.cat([pts_surf_w, pts_weight_bb],dim=0)
        occ_gt = kaolin.ops.mesh.check_sign(scan_verts[None], scan_faces, pts_d[None]).float().unsqueeze(-1)
        # # distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(pts_d[None], scan_verts[None], scan_faces)
        # # distance = torch.sqrt(distance).float().unsqueeze(-1)
        # # distance[torch.nonzero(occ_gt, as_tuple=True)] *= -1 
        if mirror:
            pts_d[:,0] *= -1
        batch['pts_d'] = pts_d.data.cpu().numpy()
        batch['occ_gt'] = occ_gt[0].data.cpu().numpy()
        # batch['weights'] = pts_w.data.cpu().numpy()
       

        np.savez(os.path.join(output_folder, 'occupancy_r.npz'), **batch)


#         # get surface normals and colors
#         # texture = TexturesUV(uv_img.unsqueeze(0), faces_uvs.unsqueeze(0), verts_uvs.unsqueeze(0))   # 用这个在渲染rgb时会报错oom
        texture = TexturesVertex(verts_features=scan_colors) 
        # meshes = Meshes(verts=[scan_verts], faces=[scan_faces], textures=texture)
        # bboxes = meshes.get_bounding_boxes().mean(2)
        # bboxes[:, 1] += 0.3
        ################ points clouds
        # verts, normals, colors = sample_points_from_meshes(meshes, num_samples=600000, return_textures=True, return_normals=True)
        # verts = verts[0] - bboxes
        # normals = normals[0]
        # colors = colors[0]
        ####################
#         # pts_fh_ret = knn_points(scan_verts.unsqueeze(0), smpl_hf_verts, K=1)
#         # pts_fh_idx = pts_fh_ret.idx[pts_fh_ret.dists <= 0.01]
#         # pts_fh = scan_verts[pts_fh_idx]
#         # num_pts_fh = pts_fh.shape[0]
#         # random_idx = torch.randint(0, num_pts_fh, [20000, 1], device=scan_verts.device)

#         batch_surf = {}
#         batch_surf['surface_points'] = verts[0].data.cpu().numpy()
#         batch_surf['surface_normals'] = normals[0].data.cpu().numpy()
#         batch_surf['surface_colors'] = colors[0].data.cpu().numpy()

#         # np.savez(os.path.join(output_folder, 'surface.npz'), **batch_surf)

        #get 2D normal maps and rgb images
        n_views = 18

        output_image_folder = os.path.join(output_folder, 'multi_view_256')
        if not os.path.exists(output_image_folder): os.makedirs(output_image_folder)

        for i in range(n_views):

        #     # rot_mat = cv2.Rodrigues(np.array([0, 2*np.pi/n_views*i, 0]))[0]
        #     # rot_mat = torch.tensor(rot_mat).cuda().float()
            rot = torch.tensor([[0, 2*np.pi/n_views*i, 0],]).float()
            # smpl_output = self.smpl(betas=torch.tensor(file['betas'][:,:10]), body_pose=torch.tensor(file['body_pose'].reshape(1,-1)).float(), global_orient=rot)
            # bboxes_mir = bboxes.clone()
            # bboxes_mir[:,0] *= -1
            # if mirror:
                # smpl_output = self.smpl(betas=torch.tensor(file['betas'][:,:10]), body_pose=torch.tensor(flipped_smpl_param[7:76]).reshape(1,-1).float(), global_orient=rot)
            # else:
            smpl_output = self.smpl(betas=torch.tensor(file['betas'][:,:10]), body_pose=torch.tensor(file['body_pose'].reshape(1,-1)).float(), global_orient=rot)
            
            smpl_tfs = smpl_output.T.clone()[0]

            # smpl_verts = smpl_output.vertices[0].cuda()
            # smpl_faces = torch.tensor(self.smpl.faces.astype(np.int32)).cuda()
            # verts_rgb = torch.zeros_like(smpl_verts).unsqueeze(0).cuda() + 0.5
            # verts_rgb[:, self.face_verts_idx] = 1
            # smpl_textures = TexturesVertex(verts_features=verts_rgb)
            # print(scan_verts_mv.shape)
            # print(scan_verts_mv[:,0].min())
            # print(scan_verts_mv[:,0].max())
            # print(scan_verts_mv[:,1].min())
            # print(scan_verts_mv[:,1].max())
            # print(scan_verts_mv[:,2].min())
            # print(scan_verts_mv[:,2].max())
            # assert False
            verts_homo = torch.ones_like(scan_verts_ori[:, 0:1])
            verts_homo = torch.cat([scan_verts_ori, verts_homo], dim=-1)
            # if debug:
            # verts_homo = torch.ones_like(smpl_verts[:, 0:1])
            # verts_homo = torch.cat([smpl_verts, verts_homo], dim=-1)
            # meshes_new = Meshes(verts=[torch.einsum('ij,nj->ni',smpl_tfs[0].cuda(),verts_homo)[:,0:3]], faces=[smpl_faces], textures=smpl_textures)
            verts = torch.einsum('ij,nj->ni',smpl_tfs[0].cuda(),verts_homo)[:,0:3]
            # verts[:, 0] = verts[:, 0] * (-1)
            meshes_new = Meshes(verts=[verts], faces=[scan_faces_ori], textures=texture)
            # flip_matrix = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().cuda()
            # flipped_vertices = torch.matmul(scan_verts_ori, flip_matrix.T)
            # meshes_new = Meshes(verts=[flipped_vertices], faces=[scan_faces_ori], textures=texture)
            # flipped_mesh = meshes_new.update_padded(new_verts=flipped_vertices)

            # points_new = Pointclouds(points=[torch.einsum('ij,nj->ni',smpl_tfs[0].cuda(),verts_homo)[:,0:3]], normals=[normals], features=[colors])

            # render normal image
            image = render_pytorch3d(meshes_new, mode='p', renderer_new=self.renderer)
            # image = render_pytorch3d_point(points_new, mode='n', renderer_new=self.renderer)
            imageio.imwrite(os.path.join(output_image_folder, '%04d_shade.png'%i), image)
            assert False
            # render rgb image
            # image_rgb = render_pytorch3d(meshes_new, mode='t', renderer_new=self.renderer)
            # print(points_new.features_packed().shape)
            # image_rgb = render_pytorch3d_point(points_new, mode='t', renderer_new=self.renderer)
            # imageio.imwrite(os.path.join(output_image_folder, '%04d_rgb.png'%i), image_rgb)
        return 

def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]

def flip_smpl_pose(pose):
    "from https://github.com/open-mmlab/mmpose/blob/0.x/mmpose/datasets/pipelines/mesh_transform.py"
    """Flip SMPL pose parameters horizontally.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
    Returns:
        pose_flipped
    """

    flippedParts = [
        0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
        20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
        38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
        59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68
    ]
    pose_flipped = pose[flippedParts]

    # Negate the second and the third dimension of the axis-angle
    pose_flipped[1::3] = -pose_flipped[1::3]
    pose_flipped[2::3] = -pose_flipped[2::3]
    return pose_flipped
    

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--scan_folder', type=str, default="/mnt/sdb/zwt/dataset/ThumanDataset/THuman2.0_Release/", help="Folder of raw scans.")
    parser.add_argument('--smpl_folder', type=str, default="./data/THuman2.0_smpl", help="Folder of fitted smpl parameters.")
    parser.add_argument('--output_folder', type=str, default="./data/THuman2.0_processed_debug", help="Output folder.")
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)

    args = parser.parse_args()

    processor = ScanProcessor()

    task = split( list(range( len(processor.scan_list))) , args.tot)[args.id]
    batch_list = []

    for i in tqdm(task):
        i = 520
        batch = processor.process(i)
    