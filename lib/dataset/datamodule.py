import os
import PIL
import torch
import hydra
import pandas
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
import torchvision.transforms as transforms

from lib.model.helpers import rectify_pose, Dict2Class

class DataSet(torch.utils.data.Dataset):

    def __init__(self, dataset_path, val=False, flip=False, opt=None):

        self.dataset_path = hydra.utils.to_absolute_path(dataset_path)

        self.cache_path = hydra.utils.to_absolute_path(opt.cache_path)
        # self.cache_path = '/mnt/sdb/zwt/gdna_addtex/outputs/gdna_addtex_coarse_new9/cache_img_dvr'
        self.cache_path = os.path.join(os.path.dirname(os.path.dirname(self.cache_path)),'cache_img_dvr')

        self.opt = opt
        self.val = val
        self.flip = flip

        self.scan_info = pandas.read_csv(hydra.utils.to_absolute_path(opt.data_list),dtype=str)

        self.n_samples = len(self.scan_info)
        print(self.n_samples)

        self.names = []
        for i in range(len(self.scan_info)):
            self.names.append(self.scan_info.iloc[i]['id'])

        if val: self.scan_info = self.scan_info[:20]

        # self.norm_transform = get_transform(self.opt.load_res)
        self.transform = get_transform(self.opt.load_res)
        # self.mask_transform = get_transform(self.opt.load_res, mask=True)
        # self.transform = get_transform(self.opt.load_res*2)
        # self.mask_transform = get_transform(self.opt.load_res*2, mask=True)

    def __getitem__(self, index):

        index = index//10

        scan_info = self.scan_info.iloc[index]

        batch = {}

        batch['index'] = index

        f = np.load(os.path.join(self.dataset_path, scan_info['id'].replace('/','_'), 'occupancy.npz') )

        batch['smpl_params'] = f['smpl_params'].astype(np.float32)
        batch['pts_d'] = f['pts_d']
        batch['occ_gt'] = f['occ_gt']
        # batch['pts_d_label'] = f['pts_label']
        # batch['pts_w'] = f['pts_w']
        
        # if self.flip:
        #     if torch.rand(1) > 0.5:
        #         batch['smpl_params'][4:76] = self.flip_smpl_pose(batch['smpl_params'][4:76])
        #         batch['pts_d'][:, 0] *= -1
        # batch['smpl_params'] = f['smpl_params'].astype(np.float32)
        batch['smpl_betas'] =  batch['smpl_params'][76:]
        batch['smpl_thetas'] = batch['smpl_params'][4:76]
        # batch['smpl_betas'] =  batch['smpl_params'][70:80] # for smplx
        # batch['smpl_thetas'] = np.concatenate((batch['smpl_params'][4:70], batch['smpl_params'][80:113]), axis=0) # for smplx
        # batch['smpl_expression'] = batch['smpl_params'][-10:]
        batch['canon_thetas'] = np.zeros_like(batch['smpl_thetas'])

        batch['scan_name'] = str(f['scan_name'])
        # batch['flip'] = batch['scan_name'][:-3] == 'mir'
        # batch['pts_d'] = f['pts_d']
        # batch['occ_gt'] = f['occ_gt']

        if self.opt.load_surface:
            surface_file = np.load(os.path.join(self.dataset_path, batch['scan_name'], 'surface.npz') )
            batch.update(surface_file)
            
        if self.opt.load_img:

            for _ in range(0, dist.get_rank()+1):
                id_view = torch.randint(low=0,high=18,size=(1,)).item()

            batch['smpl_thetas_img'] = rectify_pose(batch['smpl_thetas'].copy(), np.array([0,2*np.pi/18.*id_view,0]))
            batch['smpl_params_img'] =  batch['smpl_params'].copy()
            batch['smpl_params_img'][4:76] = batch['smpl_thetas_img']
            # batch['flip'] = batch['scan_name'][:-3] == 'mir'
            # batch['smpl_params_img'][4:70] = batch['smpl_thetas_img'][:66]
            # batch['smpl_params_img'][80:113] = batch['smpl_thetas_img'][66:]

            image_folder = os.path.join(self.dataset_path, batch['scan_name'], 'multi_view_%d'%(256))
            base_folder, _ = os.path.split(self.dataset_path)
            # segmask_path = os.path.join(base_folder, 'segmask', batch['scan_name'] + '_%04d_rgb.png'%id_view)
            batch['norm_img']= self.transform(PIL.Image.open(os.path.join(image_folder,'%04d_normal.png'%id_view)).convert('RGB'))
            batch['color_img']= self.transform(PIL.Image.open(os.path.join(image_folder[:],'%04d_rgb.png'%id_view)).convert('RGB'))
            # batch['hfmask_img'] = self.mask_transform(PIL.Image.open(segmask_path).convert('1'))

            if self.opt.load_cache:
                cache_file = np.load(os.path.join(self.cache_path, '%s.npy'%batch['scan_name']))
                # mask_face = (cache_file[:, :, 1] > 0.35) * (cache_file[:, :, 2] > -0.03)
                # cache_smpl = np.load(os.path.join(self.cache_path, '%s_pose.npy'%batch['scan_name']))
                cache_pts_d = np.load(os.path.join(self.cache_path, '%s_d.npy'%batch['scan_name']))[id_view]
                cache_vis = np.load(os.path.join(self.cache_path, '%s_vis.npy'%batch['scan_name']))[id_view]
                cache_file_vis = cache_file[:, cache_vis]
                cache_pts_d_vis = cache_pts_d[cache_vis]
                num_vis = cache_pts_d_vis.shape[0]
                rand_ind = np.random.choice(np.arange(cache_file.shape[1]), 120000-num_vis, replace=True)
                cache_file = cache_file[:, rand_ind]
                cache_pts_d = cache_pts_d[rand_ind]
                # batch['cache_pts']= cache_file[:,:,:3].reshape([-1,3])
                batch['cache_pts'] = np.concatenate([cache_file, cache_file_vis], axis=1)[:,:,:3].reshape(-1, 3)
                # mask_face = (batch['cache_pts'][:, 1] > 0.35) * (batch['cache_pts'][:, 2] > -0.03)
                # mask_hands = (batch['cache_pts'][:, 0] > 0.7) + (batch['cache_pts'][:, 0] < -0.7)

                # batch['cache_radius'] = np.ones_like(batch['cache_pts'][:, 0:1]) * 0.004
                # batch['cache_radius'][mask_face] = 0.002

                # batch['cache_radius'][mask_hands] = 0.03
                # batch['cache_mask']= cache_file[:,:,3].flatten().astype(bool)
                # batch['smpl_params_img'] = cache_smpl[id_view]
                # batch['cache_ptsd'] = cache_pts_d[id_view]
                batch['cache_ptsd'] = np.concatenate([cache_pts_d, cache_pts_d_vis], axis=0)
        return batch

    def __len__(self):

        return len(self.scan_info)*10

    def flip_smpl_pose(self, pose):
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


class DataProcessor():

    def __init__(self, opt):

        self.opt = opt
        self.total_points = 100000
        self.hands_points = 10000
        self.heads_points = 20000

    def process(self, batch, smpl_server, load_volume=True):

        num_batch,_,num_dim = batch['pts_d'].shape

        smpl_output = smpl_server(batch['smpl_params'], absolute=False)
        batch.update(smpl_output)
        # smpl_param_betas = batch['smpl_params'].clone()
        # n_batch = smpl_param_betas.shape[0]
        # if torch.rand(1) > 0.5:
        #     smpl_param_betas[:, -10] = torch.randint(-20, 20, (n_batch,)) / 10.
        # smpl_output_betas = smpl_server(smpl_param_betas, absolute=False)
        # batch.update({'smpl_verts_beta':smpl_output_betas['smpl_verts'], 'smpl_betas_train':smpl_param_betas[:, -10:]})

        if self.opt.load_img:
            
            smpl_output_img = smpl_server(batch['smpl_params_img'], absolute=False)
            smpl_output_img = { k+'_img': v for k, v in smpl_output_img.items() }
            batch.update(smpl_output_img)

        if load_volume:

            # random_idx = torch.cat([torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame//4, 1], device=batch['pts_d'].device), # 1//8 for bbox samples
            #                         torch.randint(0, self.hands_points, [num_batch, self.opt.points_per_frame//2, 1], device=batch['pts_d'].device)+self.total_points,
            #                         torch.randint(0, self.heads_points, [num_batch, self.opt.points_per_frame//2, 1], device=batch['pts_d'].device)+self.total_points+self.hands_points,
            #                         torch.randint(0 ,self.total_points, [num_batch, self.opt.points_per_frame//8, 1], device=batch['pts_d'].device)+self.total_points+self.hands_points+self.heads_points], # 1 for surface samples
            #                         1)
            random_idx = torch.cat([torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame, 1], device=batch['pts_d'].device), # 1//8 for bbox samples
                        torch.randint(0 ,self.total_points, [num_batch, self.opt.points_per_frame//8, 1], device=batch['pts_d'].device)+self.total_points], # 1 for surface samples
                        1)
            batch['occ_gt'] = torch.gather(batch['occ_gt'], 1, random_idx)
            batch['pts_d'] = torch.gather(batch['pts_d'], 1, random_idx.expand(-1, -1, num_dim))
            sum_pos = torch.sum(batch['occ_gt'], 1)
            batch['pos_weight'] = (batch['occ_gt'].shape[1] - sum_pos) / sum_pos

            # batch['pts_d_label'] = torch.gather(batch['pts_d_label'], 1, random_idx)
            # print(batch['pts_d'].shape)
            # batch['pts_w'] = torch.gather(batch['pts_w'], 1, random_idx.unsqueeze(2).expand(-1, -1, 5, 55))
            # print(batch['pts_w'].shape)
            # assert False

        if self.opt.load_surface:
            random_idx = torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame, 1], device=batch['pts_d'].device)
            batch['pts_surf'] = torch.gather(batch['surface_points'], 1, random_idx.expand(-1, -1, num_dim))
            batch['norm_surf'] = torch.gather(batch['surface_normals'], 1, random_idx.expand(-1, -1, num_dim))
            batch['color_surf'] = torch.gather(batch['surface_colors'], 1, random_idx.expand(-1, -1, num_dim))
            
        return batch

    def process_smpl(self, batch, smpl_server):

        smpl_output = smpl_server(batch['smpl_params'], absolute=False)
        
        return smpl_output

class DataModule(pl.LightningDataModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):

        # if stage == 'fit':
        self.dataset_train = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt)
        self.dataset_val = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt, val=True)
        self.meta_info = {'n_samples': self.dataset_train.n_samples,
                          'scan_info': self.dataset_train.scan_info,
                          'dataset_path': self.dataset_train.dataset_path}

        self.meta_info = Dict2Class(self.meta_info)

    def train_dataloader(self):

        dataloader = torch.utils.data.DataLoader(self.dataset_train,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=False)
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset_val,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=False,
                                pin_memory=False)
        return dataloader




def get_transform(size, mask=False):
 
    transform_list = []
    transform_list += [transforms.Resize(size)]
    transform_list += [transforms.ToTensor()]
    if not mask:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

