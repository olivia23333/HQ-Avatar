
import pytorch_lightning as pl
import hydra
import torch
import os
import numpy as np
import pandas
import json
from lib.gdna_model import BaseModel
from tqdm import trange, tqdm
from lib.model.helpers import split,rectify_pose
from lib.dataset.datamodule import DataModule, DataProcessor
from PIL import Image
from smplx import SMPL
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import time
from lib.utils.render import Renderer, render_pytorch3d_point
from pytorch3d.structures import Pointclouds
import imageio

@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(10) 

    smpl = SMPL('dataset/body_models/smpl', create_transl=False)
    scan_folder =  'dataset/ThumanDataset/THuman2.0_Release/'
    smpl_folder =  'dataset/THuman2.0_smpl/'

    datamodule = DataModule(opt.datamodule)
    datamodule.setup(stage='fit')
    meta_info = datamodule.meta_info
    data_processor = DataProcessor(opt.datamodule)

    # smpl_seg = json.load(open('/mnt/sdb/zwt/gdna_addtex/smpl_vert_segmentation.json'))
    # face_verts_idx = [i for i in smpl_seg['head'] if smpl_verts_canon[0][i][2] > -0.01]
    # hair_verts_idx = [i for i in smpl_seg['head'] if smpl_verts_canon[0][i][2] < -0.01]

    checkpoint_path = os.path.join('./checkpoints', 'last.ckpt')
    
    model = BaseModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        opt=opt.model, 
        meta_info=meta_info,
        data_processor=data_processor,
    ).cuda()

    # prepare latent codes

    batch_list = []

    output_folder = 'cache_img_dvr_512'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ## for debug
    # output_folder_debug = 'img_debug'

    task = split( list(range( meta_info.n_samples)), opt.agent_tot)[opt.agent_id]
    renderer = Renderer(512, point=True)

    for index in tqdm(task):
        scan_info = meta_info.scan_info.iloc[index]
        f = np.load(os.path.join(meta_info.dataset_path, scan_info['id'], 'occupancy.npz'))
        smpl_params = torch.tensor(f['smpl_params']).float().cuda()[None,:]
        batch = {'index': torch.tensor(index).long().cuda().reshape(1),
                'smpl_params': smpl_params,
                'scan_name': scan_info['id'],
                'smpl_betas':  smpl_params[:,76:],
                'smpl_thetas': smpl_params[:,4:76],
                'canon_thetas': torch.zeros_like(smpl_params[:,4:76]).float().cuda()
                }
        
        batch_list.append(batch)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(batch_list)):
            # batch = batch_list[193]
            batch['z_shape'] = model.z_shapes(batch['index'])            
            # batch['z_detail'] = model.z_details(batch['index'])
            batch['z_color'] = model.z_colors(batch['index'])

            batch_smpl = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, model.smpl_server)
            batch.update(batch_smpl)            
            cond = model.prepare_cond(batch)
            scan_name = batch['scan_name']

            # scan_path = os.path.join(scan_folder,scan_name, scan_name+'.obj')
            # scan_verts, faces, aux = load_obj(scan_path, device=model.device,load_textures=False)
            # scan_faces = faces.verts_idx.long()
            # pickle_path = os.path.join(smpl_folder, '%04d_smpl.pkl'%index)
            # file = pandas.read_pickle(pickle_path)
            # scan_verts = scan_verts - torch.tensor(file['transl']).cuda().float().expand(scan_verts.shape[0], -1)
            # scan_verts = scan_verts/file['scale'][0]
            # meshes = Meshes(verts=[scan_verts], faces=[scan_faces])
            # verts, _ = sample_points_from_meshes(meshes, num_samples=400000, return_normals=True)
            outputs_list = []
            smpl_param_list = []
            pts_d_list = []

            ## for debug
            # debug_dir = os.path.join(output_folder_debug, scan_name)
            # if not os.path.exists(debug_dir):
            #     os.makedirs(debug_dir)

            smpl_params = batch['smpl_params'][0].data.cpu().numpy()

            # n = 18
            # for k in trange(n):
            #     smpl_thetas = rectify_pose(smpl_params[4:76], np.array([0,2*np.pi/n*k,0]))
            #     smpl_params[4:76] = smpl_thetas
            #     smpl_param_list.append(smpl_params.copy())

            # smpl_thetas = rectify_pose(smpl_params[4:76], np.array([0,2*np.pi/n*k,0]))
            # smpl_thetas = rectify_pose(smpl_params[4:76], np.array([0,0.,0]))
            
            # smpl_params[4:76] = smpl_thetas
            # smpl_param_list.append(smpl_params.copy())
            param_canonical = torch.zeros((1, 86),dtype=torch.float32).cuda()
            param_canonical[0, 0] = 1
            param_canonical[0, 9] = np.pi / 6
            param_canonical[0, 12] = -np.pi / 6
            param_canonical[0, -10:] = torch.tensor(smpl_params[-10:]).cuda().float()

            smpl_output = model.smpl_server(param_canonical, absolute=False)
            start_time = time.time()
            pts_c, weights = model.extract_points(smpl_output['smpl_tfs'], 
                                        smpl_output['smpl_verts'], 
                                        cond, 
                                        eval_mode=True, 
                                        fine=False,
                                        num_points=1000000,
                                        init_points=None)
            smpl_output = model.smpl_server(torch.tensor(smpl_params[None]).cuda().float(), absolute=False)
            pts_d = model.deform_point(pts_c, weights, smpl_output['smpl_tfs'])
            # norms = torch.ones_like(pts_d[0])
            # colors = torch.ones_like(pts_d[0]) * 0.5
            # points_new = Pointclouds(points=[pts_d[0]], normals=[norms], features=[colors])
            # abc, visible = render_pytorch3d_point(points_new, mode='t', renderer_new=renderer)
            # imageio.imwrite('/mnt/sdb/zwt/gdna_addtex/debug/debug.png', abc)
            # assert False
            
            # point_mask, pts_d = model.forward_points(smpl_output['smpl_tfs'], 
            #                 smpl_output['smpl_verts'], 
            #                 cond, 
            #                 eval_mode=True, 
            #                 fine=False,
            #                 num_points=500000,
            #                 init_points=point_init,
            #                 radius=0.01,
            #                 use_init=True)
            # end_time = time.time()
            # print(end_time-start_time)
            # for con in cond:
            #     if con == 'latent':
            #         print(cond[con][0].shape)
            #         print(cond[con][1].shape)
            #     else:
            #         print(cond[con].shape)
            # assert False
            # point_mask, _ = model.forward_points(smpl_output['smpl_tfs'], 
            #                 smpl_output['smpl_verts'], 
            #                 cond, 
            #                 eval_mode=True, 
            #                 fine=False,
            #                 num_points=1000000,
            #                 init_points=point_init,
            #                 radius=0.01,
            #                 use_init=True)
            pts_d = pts_d[0]
            n = 18
            for k in range(n):
                # smpl_thetas = rectify_pose(smpl_params[4:76], np.array([0,2*np.pi/n*k,0]))
                # smpl_params[4:76] = smpl_thetas
                rot = torch.tensor([[0, 2*np.pi/n*k, 0],]).float()
                smpl_param_list.append(smpl_params.copy())
                smpl_output = smpl(betas=torch.tensor(smpl_params[-10:]).unsqueeze(0), body_pose=torch.tensor(smpl_params[7:76]).unsqueeze(0).float(), global_orient=rot)
                smpl_tfs = smpl_output.T.clone()[0]
                pts_homo = torch.ones_like(pts_d[:, 0:1])
                pts_homo = torch.cat([pts_d, pts_homo], dim=-1)
                pts_d_n = torch.einsum('ij,nj->ni',smpl_tfs[0].cuda(), pts_homo)[:,0:3]
                norms = torch.ones_like(pts_d_n)
                colors = torch.ones_like(pts_d_n) * 0.5
                points_new = Pointclouds(points=[pts_d_n], normals=[norms], features=[colors])
                _, visible = render_pytorch3d_point(points_new, mode='t', renderer_new=renderer)
                pts_d_n = pts_d_n.data.cpu().numpy()
                # pts_d_list.append(verts[0].data.cpu().numpy())
                pts_d_list.append(pts_d_n)
            # random_idx = torch.randint(0, point_mask.shape[1], [1, 800000, 1], device=point_mask.device)
            # point_mask = torch.gather(point_mask, 1, random_idx.expand(-1, -1, 4))
                # img_mask = model.volume_render(smpl_output['smpl_tfs'], 
                #                             smpl_output['smpl_verts'], 
                #                             cond, 
                #                             eval_mode=True, 
                #                             fine=False,
                #                             res=512)
                # assert False
                
                # mesh_cano = model.extract_mesh(smpl_output['smpl_verts_cano'], smpl_output['smpl_tfs'], cond, res_up=4)
                # mesh_def = model.deform_mesh(mesh_cano, smpl_output['smpl_tfs'])
                # img, _, mask, _, _, _ = model.forward_image_diff(mesh_def['verts'], mesh_def['faces'])
                # assert False
            
                # outputs_list.append(img_mask)

                ## for debug
                # img = img_mask[:,:,0:3]
                # img = Image.fromarray(np.uint8(255*(img*0.5 + 0.5)))
                # img.save(os.path.join(debug_dir, str(k)+'.png'))
            
            # assert False
            # outputs_all = np.stack(outputs_list, axis=0)
            smpl_all = np.stack(smpl_param_list, axis=0)
            pts_all = np.stack(pts_d_list, axis=0)

            # np.save(os.path.join(output_folder,'%s.npy'%scan_name),point_mask)
            np.save(os.path.join(output_folder,'%s_d.npy'%scan_name),pts_all)
            np.save(os.path.join(output_folder,'%s_pose.npy'%scan_name),smpl_all)


if __name__ == '__main__':
    main()