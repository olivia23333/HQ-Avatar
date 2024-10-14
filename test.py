
import pytorch_lightning as pl
import hydra
import torch
import os
import numpy as np
from lib.gdna_model import BaseModel
from tqdm import tqdm
import imageio
from lib.utils.render import render_mesh_dict, Renderer
import glob
from lib.dataset.datamodule import DataProcessor
from lib.model.helpers import Dict2Class
import pandas
from lib.model.helpers import split,rectify_pose

@hydra.main(config_path="config", config_name="config")

def main(opt):

    print(opt.pretty())
    pl.seed_everything(opt.seed, workers=True)
    torch.set_num_threads(10) 

    scan_info = pandas.read_csv(hydra.utils.to_absolute_path(opt.datamodule.data_list))
    meta_info = Dict2Class({'n_samples': len(scan_info)})

    data_processor = DataProcessor(opt.datamodule)
    checkpoint_path = os.path.join('./checkpoints', 'last.ckpt')
    
    model = BaseModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        opt=opt.model, 
        meta_info=meta_info,
        data_processor=data_processor,
    ).cuda()


    renderer = Renderer(1024, anti_alias=False, real_cam=False)
    # renderer = Renderer(256, anti_alias=True)

    output_folder = 'results_normal'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    max_samples = 200

    smpl_param_zero = torch.zeros((1,86)).cuda().float()
    smpl_param_zero[:,0] = 1

    motion_folder = hydra.utils.to_absolute_path('data/aist_demo/seqs')
    motion_files = sorted(glob.glob(os.path.join(motion_folder, '*.npz')))
    smpl_param_anim = []
    for f in motion_files:
        f = np.load(f)
        smpl_params = np.zeros(86)
        smpl_params[0], smpl_params[4:76] = 1, f['pose']
        smpl_param_anim.append(torch.tensor(smpl_params))
    smpl_param_anim = torch.stack(smpl_param_anim).float().cuda()

    # pickle_path = '/mnt/sdb/zwt/gdna_addtex/data/THUman20_Smpl-X/0007/smplx_param.pkl'
    # file = pandas.read_pickle(pickle_path)
    # smplx_param_sample = np.concatenate([np.ones( (1,1)), 
    #                     np.zeros( (1,3)),
    #                     file['global_orient'].reshape(1,-1),
    #                     file['body_pose'].reshape(1,-1),
    #                     file['betas'][:,:10],
    #                     file['left_hand_pose'].reshape(1,-1),
    #                     file['right_hand_pose'].reshape(1,-1),
    #                     file['jaw_pose'].reshape(1,-1),
    #                     file['leye_pose'].reshape(1,-1),
    #                     file['reye_pose'].reshape(1,-1),
    #                     file['expression'].reshape(1,-1),], axis=1)

    n_views = 90
    smpl_param_round = []
    for i in range(n_views+1):
        rot = np.array([0, 2*np.pi/n_views*i, 0])
        smpl_params = np.zeros(86)
        # smpl_params = np.zeros(123)
        # smpl_params = smpl_param_zero
        smpl_params[0] = 1
        smpl_params[4:7] = rot
        smpl_param_round.append(torch.tensor(smpl_params))
    smpl_param_round = torch.stack(smpl_param_round).float().cuda()
    # smplx_param_sample = torch.tensor(smplx_param_sample).float().cuda()

    batch_list = []
    # prepare latent codes
    if opt.eval_mode == 'z_shape':

        idx_b = np.random.randint(0, meta_info.n_samples)
        
        # while len(batch_list) < max_samples:
        while len(batch_list) < 1:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_shape_a = model.z_shapes.weight.data[idx_a]
            z_shape_b = model.z_shapes.weight.data[idx_b]
            z_detail = model.z_details.weight.data.mean(0)
            z_color = model.z_colors.weight.data.mean(0)
            # z_color = model.z_colors.weight.data[idx_a]

            for i in range(10):

                z_shape = torch.lerp(z_shape_a, z_shape_b, i/10)

                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'z_color': z_color[None],
                        'smpl_params': smpl_param_zero,
                        'canon_thetas': smpl_param_zero[:, 4:76],
                        'smpl_betas': smpl_param_zero[:, -10:],
                        'smpl_thetas': smpl_param_zero[:, 4:76]}

                batch_list.append(batch)

    
    if opt.eval_mode == 'z_detail':
        idx_b = np.random.randint(0, meta_info.n_samples)
        
        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_detail_a = model.z_details.weight.data[idx_a]
            z_detail_b = model.z_details.weight.data[idx_b]
            z_shape = model.z_shapes.weight.data.mean(0)
            z_color = model.z_colors.weight.data.mean(0)

            for i in range(10):

                z_detail = torch.lerp(z_detail_a, z_detail_b, i/10)


                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'z_color': z_color[None],
                        'smpl_params': smpl_param_zero}

                batch_list.append(batch)


    if opt.eval_mode == 'z_color':
        idx_b = np.random.randint(0, meta_info.n_samples)
        
        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_color_a = model.z_colors.weight.data[idx_a]
            z_color_b = model.z_colors.weight.data[idx_b]
            z_shape = model.z_shapes.weight.data.mean(0)
            z_detail = model.z_details.weight.data.mean(0)

            for i in range(10):

                z_color = torch.lerp(z_color_a, z_color_b, i/10)


                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'z_color': z_color[None],
                        'smpl_params': smpl_param_zero,
                        'canon_thetas': smpl_param_zero[:, 4:76]}

                batch_list.append(batch)


    if opt.eval_mode == 'betas':
        z_shape = model.z_shapes.weight.data.mean(0)
        z_detail = model.z_details.weight.data.mean(0)
        z_color = model.z_colors.weight.data.mean(0)
        # z_shape = model.z_shapes.weight.data[0]
        # z_detail = model.z_details.weight.data[0]
        # z_color = model.z_colors.weight.data[0]

        betas = torch.cat([torch.linspace(0,-2,10),
                            torch.linspace(-2,0,10),
                            torch.linspace(0,2,10),
                            torch.linspace(2,0,10)])

        for i in range(len(betas)):
            smpl_param = smpl_param_zero.clone()
            smpl_param[:, -10] = betas[i]

            batch = {'z_shape': z_shape[None],
                    'z_detail': z_detail[None],
                    'z_color': z_color[None],
                    'smpl_params': smpl_param,
                    'canon_thetas': smpl_param[:, 4:76],
                    'smpl_betas': smpl_param[:, -10:],
                    'smpl_thetas': smpl_param[:, 4:76]}

            batch_list.append(batch)

    if opt.eval_mode == 'thetas':
        # z_shape = model.z_shapes.weight.data.mean(0)
        # z_detail = model.z_details.weight.data.mean(0)
        # z_color = model.z_colors.weight.data.mean(0)
        z_shape = model.z_shapes.weight.data[21]
        z_detail = model.z_details.weight.data[21]
        z_color = model.z_colors.weight.data[21]

        for i in range(len(smpl_param_anim)):
            batch = {'z_shape': z_shape[None],
                     'z_detail': z_detail[None],
                     'z_color': z_color[None],
                     'smpl_params': smpl_param_anim[[i]],
                     'canon_thetas': smpl_param_zero[:, 4:76],
                     'smpl_betas': smpl_param_anim[[i]][:, -10:],
                     'smpl_thetas': smpl_param_anim[[i]][:, 4:76]
                    }

            batch_list.append(batch)
    
    if opt.eval_mode == 'view':
        # z_shape = model.z_shapes.weight.data.mean(0)
        # z_detail = model.z_details.weight.data.mean(0)
        # z_color = model.z_colors.weight.data.mean(0)
        # z_shape = model.z_shapes.weight.data[100]
        # z_detail = model.z_details.weight.data[100]
        # z_color = model.z_colors.weight.data[100]
        for i in range(396):
            batch = {'z_shape': model.z_shapes.weight.data[i][None],
                    #  'z_detail': model.z_details.weight.data[i][None],
                     'z_color': model.z_colors.weight.data[i][None],
                     'smpl_params': smpl_param_zero,
                     'canon_thetas': smpl_param_zero[:, 4:76],
                     'smpl_betas': smpl_param_zero[:, -10:],
                     'smpl_thetas': smpl_param_zero[:, 4:76]
                    #  'smpl_expression': smpl_param_zero[:,-10:]
                    }

            batch_list.append(batch)
    
    if opt.eval_mode == 'demos':
        idx_a = np.random.randint(0, meta_info.n_samples)
        idx_b = np.random.randint(0, meta_info.n_samples)
        z_shape = (model.z_shapes.weight.data[idx_a] + model.z_shapes.weight.data[idx_b])
        z_detail = (model.z_details.weight.data[idx_a] + model.z_details.weight.data[idx_b])
        z_color = (model.z_colors.weight.data[idx_a] + model.z_colors.weight.data[idx_b])
        for i in range(len(smpl_param_anim)):
            batch = {'z_shape': z_shape[None],
                     'z_detail': z_detail[None],
                     'z_color': z_color[None],
                     'smpl_params': smpl_param_anim[[i]]
                    }

            batch_list.append(batch)

    if opt.eval_mode == 'sample':

        z_shapes, z_colors = model.sample_codes(max_samples)

        for i in range(len(z_shapes)):

            id_smpl = np.random.randint(len(smpl_param_anim))
            batch = {'z_shape': z_shapes[i][None],
                    # 'z_detail': z_details[i][None],
                    'z_color': z_colors[i][None],
                    'smpl_params': smpl_param_anim[id_smpl][None],
                    'smpl_betas': smpl_param_anim[[i]][:, -10:],
                    'smpl_thetas': smpl_param_zero[:, 4:76],
                    'canon_thetas': smpl_param_zero[:, 4:76],
                    }
            
            batch_list.append(batch)
    
    if opt.eval_mode == 'surround':

        z_shape = model.z_shapes.weight.data.mean(0)
        z_detail = model.z_details.weight.data.mean(0)
        z_color = model.z_colors.weight.data.mean(0)
        # z_shape = model.z_shapes.weight.data[7]
        # z_detail = model.z_details.weight.data[7]
        # z_color = model.z_colors.weight.data[7]
       
        for i in range(len(smpl_param_round)):
            batch = {'z_shape': z_shape[None],
                     'z_detail': z_detail[None],
                     'z_color': z_color[None],
                     'smpl_params': smpl_param_round[[i]],
                     'canon_thetas': smpl_param_zero[:, 4:76],
                     'smpl_betas': smpl_param_round[[i]][:,70:80],
                     'smpl_thetas': torch.cat((smpl_param_round[[i]][:,4:70], smpl_param_round[[i]][:,80:113]), dim=1),
                     'smpl_expression': smpl_param_round[[i]][:,-10:],
                    }
            batch_list.append(batch)

    if opt.eval_mode == 'interp':


        idx_b = np.random.randint(0, meta_info.n_samples)
        
        while len(batch_list) < max_samples:
            idx_a = idx_b
            idx_b = np.random.randint(0, meta_info.n_samples)

            z_shape_a = model.z_shapes.weight.data[idx_a]
            z_shape_b = model.z_shapes.weight.data[idx_b]

            z_detail_a = model.z_details.weight.data[idx_a]
            z_detail_b = model.z_details.weight.data[idx_b]

            z_color_a = model.z_colors.weight.data[idx_a]
            z_color_b = model.z_colors.weight.data[idx_b]

            for i in range(10):

                z_shape = torch.lerp(z_shape_a, z_shape_b, i/10)
                z_detail = torch.lerp(z_detail_a, z_detail_b, i/10)
                z_color = torch.lerp(z_color_a, z_color_b, i/10)

                batch = {'z_shape': z_shape[None],
                        'z_detail': z_detail[None],
                        'z_color': z_color[None],
                        'smpl_params': smpl_param_anim[len(batch_list)][None]}

                batch_list.append(batch)


    images_all = []
    smplfolder = '/mnt/sdb/zwt/gdna_addtex/data/THuman2.0_smpl'
    pickle_path = os.path.join(smplfolder, '0520_smpl.pkl')
    file = pandas.read_pickle(pickle_path)
    # print(file['betas'].shape)
    # assert False
    smpl_params = np.concatenate([file['global_orient'].reshape(1, -1), file['body_pose'].reshape(1, -1)], axis=1)
    smpl_params = torch.tensor(smpl_params).float().cuda()
    # smpl_tfs_cano = torch.eye(4).expand(24,4,4).unsqueeze(0)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(batch_list)):
            # n = 18
            # for k in range(n):
        
            cond = model.prepare_cond(batch, train=False)
            batch_smpl = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, model.smpl_server)
            # smpl_tfs_cano = torch.eye(4).expand(24,4,4).unsqueeze(0)
            mesh_cano = model.extract_mesh(batch_smpl['smpl_verts_cano'], batch_smpl['smpl_tfs'], cond, res_up=4)
            # mesh_cano = model.extract_mesh(batch_smpl['smpl_verts_cano'], smpl_tfs_cano, cond, res_up=4)
            mesh_def = model.deform_mesh(mesh_cano, batch_smpl['smpl_tfs'])

            # img_def = render_mesh_dict(mesh_def,mode='xy',render_new=renderer)
            img_def = render_mesh_dict(mesh_def,mode='xy',render_new=renderer)
            # img_def = np.concatenate([img_def[:256,:,:], img_def[256:,:,:]],axis=1)
            # img_def = np.concatenate([img_def[:1024,:,:], img_def[1024:,:,:]],axis=1)
            imageio.imwrite(os.path.join(output_folder, '%04d_norm.png'%i), img_def[:1024,:,:])
            imageio.imwrite(os.path.join(output_folder, '%04d_rgb.png'%i), img_def[1024:,:,:])

            images_all.append(img_def)

            if i%9 == 0:
                imageio.mimsave(os.path.join(output_folder,'%s_seed%d.mp4' % (opt.eval_mode,opt.seed)),images_all, codec='libx264')
            
            # cond = model.prepare_cond(batch, train=False)
            # # smpl_thetas = rectify_pose(batch['smpl_params'][0, 4:76].data.cpu().numpy(), np.array([0,0.,0]))
            # smpl_thetas = rectify_pose(smpl_params[0].data.cpu().numpy(), np.array([0,0.,0]))
            # batch['smpl_params'][0, 4:76] = torch.tensor(smpl_thetas).float().cuda()
            # smpl_output = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, model.smpl_server)
            # point_mask_initial, point_init = model.forward_points(smpl_output['smpl_tfs'], 
            #                 smpl_output['smpl_verts'], 
            #                 cond, 
            #                 eval_mode=True, 
            #                 fine=False,
            #                 num_points=110240,
            #                 init_points=None)

            # point_mask, pts_d = model.forward_points(smpl_output['smpl_tfs'], 
            #     smpl_output['smpl_verts'], 
            #     cond, 
            #     eval_mode=True, 
            #     fine=False,
            #     num_points=1000000,
            #     init_points=point_init,
            #     radius=0.01,
            #     use_init=True)

            # # points = torch.tensor(point_mask[:,:,:3]).float().cuda()
            # points = torch.tensor(point_mask_initial[:,:,:3]).float().cuda()
            # images_norm, images_rgb = model.evel_pointrendering(points, cond, smpl_output['smpl_tfs'], point_init)
            # images_norm = images_norm[0].permute(1, 2, 0).data.cpu().numpy()
            # images_rgb = images_rgb[0].permute(1, 2, 0).data.cpu().numpy()
            # imageio.imwrite(os.path.join(output_folder, '%04d_norm.png'%(i)), images_norm)
            # imageio.imwrite(os.path.join(output_folder, '%04d_rgb.png'%(i)), images_rgb)
            # assert False
                

if __name__ == '__main__':
    main()