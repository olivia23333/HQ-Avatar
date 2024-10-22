import numpy as np
import torchvision
import torch

import torch.nn.functional as F
import cv2
import pytorch3d.ops as ops

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def query_weights_smpl(x, smpl_verts, smpl_weights):
    
    distance_batch, index_batch, neighbor_points  = ops.knn_points(x,smpl_verts,K=1,return_nn=True)

    index_batch = index_batch[0]

    skinning_weights = smpl_weights[:,index_batch][:,:,0,:]

    return skinning_weights

def create_voxel_grid(d, h, w):
    x_range = (torch.linspace(-1,1,steps=w)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

def bmv(m, v):
    return (m*v.transpose(-1,-2).expand(-1,3,-1)).sum(-1,keepdim=True)

def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]

def vis_images(batch):

    images = []
    for key in batch:
        if key != 'hf_mask':
            img = torchvision.utils.make_grid( batch[key], normalize=True, range=(-1,1), nrow=8).permute(1,2,0).data.cpu().numpy()
            images.append(img)
    return np.concatenate(images, axis=0)
    
def select_dict(dict, keys):
    return {key:dict[key] for key in dict if key in keys}

def mask_dict(dict, mask):

    dict_new = {}
    for key in dict:
        dict_new[key] = dict[key][mask]

    return dict_new

def index_dict(dict, start, end):

    for key in dict:
        dict[key] = dict[key][start:end]

    return dict


def grid_sample_feat(feat_maps, x, plane_axes):
    
    n_batch, n_point, _ = x.shape
    
    if feat_maps[0].ndim == 4:
        # x = x[:,:,None,:2]
        # assert False
        feats = []
        for feat_map in feat_maps:
            feat_map = feat_map.view(n_batch, 3, -1, feat_map.shape[-2], feat_map.shape[-1])
            feat_map = feat_map.view(n_batch*3, -1, feat_map.shape[-2], feat_map.shape[-1])
            projected_x = project_onto_planes(plane_axes, x).unsqueeze(1)
            feats.append(F.grid_sample(feat_map, projected_x, mode='bilinear', padding_mode='zeros', align_corners=True).permute(0, 3, 2, 1).reshape(n_batch, 3, n_point, -1))
        # feats = feats.mean(1)
            # print(feat_map.shape)
        feats = torch.cat(feats, -1).mean(1)
        # print(feats.shape)
     
    elif feat_maps[0].ndim == 5:
        x = x[:,:,None,None,:3]

        feats = F.grid_sample(feat_maps, x, align_corners=True, mode='bilinear',padding_mode='zeros')
        feats = feats.reshape(n_batch, -1, n_point).transpose(1,2)
        # print(feats.shape)
        # assert False
    return feats

def project_onto_planes(planes, coordinates):
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = planes.unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3).to(coordinates.device)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def expand_cond(cond, x):

    cond = cond[:, None]
    new_shape = list(cond.shape)
    new_shape[0] = x.shape[0]
    new_shape[1] = x.shape[1]
    
    return cond.expand(new_shape)

def rectify_pose(pose, rot):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_rot = cv2.Rodrigues(rot)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    # new_root = np.linalg.inv(R_abs).dot(R_root)
    new_root = R_rot.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose


class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
  
