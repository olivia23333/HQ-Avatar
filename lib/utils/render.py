import cv2
import torch
import numpy as np
import math
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    PointsRenderer,
    MeshRasterizer,
    PointsRasterizer,
    HardPhongShader,
    SoftPhongShader,
    AlphaCompositor,
    PointLights
)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh import Textures

class Renderer():
    def __init__(self, image_size=256,anti_alias=False,real_cam=False, point=False, R=None, device=None):
        super().__init__()

        self.anti_alias = anti_alias

        self.image_size = image_size

        if device == None:
            self.device = torch.device("cuda:0")
        else:
            self.device = device
        torch.cuda.set_device(self.device)
        if R == None:
            R = torch.from_numpy(np.array([[-1., 0., 0.],
                                        [0., 1., 0.],
                                        [0., 0., -1.]])).float().unsqueeze(0).to(self.device)

        t = torch.from_numpy(np.array([[0., 0.3, 2.]])).float().to(self.device)

        self.R = R

        if real_cam:
            self.cameras = FoVPerspectiveCameras(R=R, T=t, fov=60, device=self.device)
        else:
            self.cameras = FoVOrthographicCameras(R=R, T=t, device=self.device)
        # print(self.cameras.get_camera_center())
        # assert False

        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        if anti_alias: image_size = image_size*2
        if point:
            self.raster_settings = PointsRasterizationSettings(image_size=image_size,
                # radius=0.3 * (0.75 ** math.log2(600000 / 100)),
                radius=0.003,
                # radius=0.003,
                points_per_pixel=10
            )
            self.rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
            self.compositor = AlphaCompositor(background_color=[1, 1, 1, 0])
            # self.compositor = AlphaCompositor().to(self.device)
            self.renderer = PointsRenderer(rasterizer=self.rasterizer, compositor=self.compositor)
        else:
            self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=100,blur_radius=0,perspective_correct=True)
            self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

            self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

            self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_dict(self, mesh_dict, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():


            if 'norm' not in mesh_dict:
                mesh = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None])
                normals = torch.stack(mesh.verts_normals_list())
            else:
                normals = mesh_dict['norm'][None]

            front_light = torch.tensor([0,0,1]).float().to(mesh_dict['verts'].device)
            mesh = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None])
            normals_coarse = torch.stack(mesh.verts_normals_list())
            shades = (normals_coarse * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            # shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            # shades = abs((normals * front_light.view(1,1,3)).sum(-1)).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'x' in mode or 'y' in mode:
                mesh = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None])
                normals_coarse = torch.stack(mesh.verts_normals_list())[0]
                normals_fine = mesh_dict['norm']
                # normals_fine = 1    ## 为了给coarse模型生成test展示视频 

                cos_dis = (normals_coarse*normals_fine).sum(1, keepdims=True)  # [-1,1]

                sigma = 0.5
                fine_confidence = 0.5*(cos_dis + 1) # 0~1
                fine_confidence = torch.exp(-(fine_confidence-1)**2/2.0/sigma/sigma)

                fused_n = normals_fine*fine_confidence + normals_coarse*(1-fine_confidence)  # 融合coarse shape的normal和fine网络学出的normal
                normals_x = fused_n / ((fused_n**2).sum(1, keepdims=True))**0.5
                normals_x = normals_x[None]
                shades_x = (normals_x * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                # normals_vis = (normals.clamp(-1, 1) + 1) / 2
                mesh_normal = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=normals_vis))   # 即让mesh的texture(每个点的color)为normal值
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'p' in mode:
                mesh_shading = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=shades))    # 让mesh的texture为shading
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'y' in mode:
                normals_vis_x = normals_x* 0.5 + 0.5  # 归一化到[0,1]
                mesh_normal = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=normals_vis_x))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'x' in mode:
                mesh_shading = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=shades_x))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'a' in mode:    # albedo  即不加光照只有texture
                assert('color' in mesh_dict)
                mesh_albido = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=mesh_dict['color'][None]))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            # if 't' in mode:
            #     assert('color' in mesh_dict)
            #     mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=mesh_dict['color'][None]*shades))
            #     image_color = self.renderer(mesh_teture)
            #     results.append(image_color)

            # mine
            if 't' in mode:    # texture  加上光照 （这里表述为texture=albedo+illumination） 应该用这个，因为tex_network输出的是不带光照的纹理（albedo），因为读取数据时读的是albedo（只读了uv图，没读.mtl中的光照信息）
                assert('color' in mesh_dict)
                # color_vis = mesh_dict['color'][None] * 0.5 + 0.5    # 反归一化 (-1,1)->(0,1)
                color_vis = (mesh_dict['color'][None][:,:,:3].clamp(-1, 1) + 1) / 2
                # mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=color_vis*shades))
                mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=color_vis))    # 不加shading了  shading只是为了展示mesh更加立体随便加的阴影，不是真正的光照
                image_color = self.renderer(mesh_teture)
                results.append(image_color)


            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
            return  results
    # def render_point_dict(self, point_dict, mode='npat'):
    def render_point_dict(self, points, color=True):
        '''
        mode: normal, phong, texture
        '''
        # with torch.no_grad():

            # normals = point_dict['norm'][None]
            # front_light = torch.tensor([0,0,1]).float().to(point_dict['points'].device)
            # shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            # shades = abs((normals * front_light.view(1,1,3)).sum(-1)).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            # results = []

            # if 'n' in mode:
            #     normals_vis = normals* 0.5 + 0.5 
            #     # mesh_normal = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=normals_vis))   # 即让mesh的texture(每个点的color)为normal值
            #     points_normal = Pointclouds(points=point_dict['points'][None], normals=normals, features=normals_vis)
            #     image_normal = self.renderer(points_normal)
            #     results.append(image_normal)

            # if 'p' in mode:
            #     points_shading = Pointclouds(points=point_dict['points'][None], normals=normals, features=shades)
            #     # mesh_shading = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=shades))    # 让mesh的texture为shading
            #     image_phong = self.renderer(points_shading)
            #     results.append(image_phong)

            # mine
            # if 't' in mode:    # texture  加上光照 （这里表述为texture=albedo+illumination） 应该用这个，因为tex_network输出的是不带光照的纹理（albedo），因为读取数据时读的是albedo（只读了uv图，没读.mtl中的光照信息）
                # assert('color' in point_dict)
                # color_vis = point_dict['color'][None] * 0.5 + 0.5    # 反归一化 (-1,1)->(0,1)
        fragments = self.rasterizer(points)
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        if color:
            alphas = 1 - dists2 / (r * r)
            image_color, _ = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                points.features_packed().permute(1, 0),
            )
        else:
            render_index = fragments.idx.long().flatten()
            mask = render_index == -1
            render_index[mask] = 0
            sigmas = torch.gather(points.features_packed()[:,0], 0, render_index)
            sigmas[mask] = 0
            sigmas = sigmas.reshape(fragments.idx.shape).permute(0, 3, 1, 2)
            # alphas = 1 - dists2 / (r * r)
            alphas = (1 - dists2 / (r * r)) * sigmas
            # image_color, _ = self.compositor(
            #     fragments.idx.long().permute(0, 3, 1, 2),
            #     alphas,
            #     points.features_packed().permute(1, 0),
            # )
            # image_color, _ = self.compositor(
            #     fragments.idx.long().permute(0, 3, 1, 2),
            #     alphas,
            #     points.features_packed().permute(1, 0),
            # )
            feat = torch.ones_like(points.features_packed().permute(1, 0))
            image_color, _ = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                alphas,
                feat,
            )
            # image_color = torch.sum(image_color, dim=1, keepdim=True)
                # mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=color_vis*shades))
                # mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=color_vis))    # 不加shading了  shading只是为了展示mesh更加立体随便加的阴影，不是真正的光照
                # points_teture = Pointclouds(points=point_dict['points'][None], normals=normals, features=color_vis)
                # image_color = self.renderer(points_teture)
                # results.append(image_color)


            # results = torch.cat(results, axis=1)

            # if self.anti_alias:
            #     results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
            #     results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
            #     results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
        return  image_color



    def render_mesh(self, verts, faces, colors=None, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = torch.tensor([0,0,1]).float().to(verts.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'a' in mode: 
                assert(colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            if 't' in mode: 
                assert(colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors*shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)


            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
            return  results


    def render_mesh_pytorch(self, mesh, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            normals = torch.stack(mesh.verts_normals_list())
            # for mirror
            # normals[:,:,1:] *= -1
            front_light = torch.tensor([0,0,1]).float().to(mesh.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = mesh.clone()
                mesh_normal.textures = Textures(verts_rgb=normals_vis)
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            if 'p' in mode:
                mesh_shading = mesh.clone()
                mesh_shading.textures = Textures(verts_rgb=shades)
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            if 'a' in mode: 
                image_color = self.renderer(mesh)
                results.append(image_color)
            
            if 't' in mode: 
                image_color = self.renderer(mesh)
                results.append(image_color)


            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC                    
            
            return  results

    def render_point_pytorch(self, points, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            normals = points.estimate_normals(neighborhood_size=1500)
            normals = normals * -1
            # print(normals.shape)
            # normals = torch.stack(points.normals_list())
            # print(normals.shape)
            # assert False
            front_light = torch.tensor([0,0,1]).float().to(points.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            # shades = abs((normals * front_light.view(1,1,3)).sum(-1)).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                points_normal = points.clone()
                points_normal.features = normals_vis
                image_normal = self.renderer(points_normal)
                results.append(image_normal)

            if 'p' in mode:
                points_shading = points.clone()
                points_shading.features = shades
                # points_shading = Pointclouds(points=point_dict['points'][None], normals=normals, features=shades)
                # mesh_shading = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=shades))    # 让mesh的texture为shading
                image_phong = self.renderer(points_shading)
                results.append(image_phong)

            # mine
            if 't' in mode:    # texture  加上光照 （这里表述为texture=albedo+illumination） 应该用这个，因为tex_network输出的是不带光照的纹理（albedo），因为读取数据时读的是albedo（只读了uv图，没读.mtl中的光照信息）
                # assert('color' in point_dict)
                # color_vis = point_dict['color'][None] * 0.5 + 0.5    # 反归一化 (-1,1)->(0,1)
                # mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=color_vis*shades))
                # mesh_teture = Meshes(mesh_dict['verts'][None], mesh_dict['faces'][None], textures=Textures(verts_rgb=color_vis))    # 不加shading了  shading只是为了展示mesh更加立体随便加的阴影，不是真正的光照
                # points_teture = Pointclouds(points=point_dict['points'][None], normals=normals, features=color_vis)
                normals_vis = normals* 0.5 + 0.5 
                n_points = points._P
                visible = torch.zeros(n_points).bool().to(points.device)
                fragments = self.rasterizer(points)
                r = self.rasterizer.raster_settings.radius
                dists2 = fragments.dists.permute(0, 3, 1, 2)
                alphas = 1 - dists2 / (r * r)
                
                image_color, weights = self.compositor(
                    fragments.idx.long().permute(0, 3, 1, 2),
                    alphas,
                    points.features_packed().permute(1, 0),
                )
                # shades = torch.cat([shades, torch.ones_like(shades)[:,:,:1]], dim=-1)
                # image_color, weights = self.compositor(
                #     fragments.idx.long().permute(0, 3, 1, 2),
                #     alphas,
                #     shades[0].permute(1, 0),
                # )
                # print(image_color.shape)
                # assert False
                # image_color = self.renderer(points)
                # print(image_color.shape)
                # print(visible.sum())
                results.append(image_color.permute(0, 2, 3, 1))
                visible_points = fragments.idx.long()[..., 0].reshape(-1)
                visible_points = visible_points[visible_points != -1]
                visible_points = visible_points % n_points
                visible[visible_points] = True
                # print(visible.sum())

                visible_points = fragments.idx.long().reshape(-1)[weights.reshape(-1) > 0.5]
                visible_points = visible_points[visible_points != -1]
                visible_points = visible_points % n_points
                visible[visible_points] = True
                # print(visible.sum())



            results = torch.cat(results, axis=1)

            if self.anti_alias:
                results = results.permute(0, 3, 1, 2)  # NHWC -> NCHW
                results = torch.nn.functional.interpolate(results, scale_factor=0.5,mode='bilinear',align_corners=True)
                results = results.permute(0, 2, 3, 1)  # NCHW -> NHWC  

            if 't' in mode:
                return  results, visible
            else:
                return results


renderer = Renderer(image_size=1024, real_cam=False)
def render(verts, faces, colors=None):
    return renderer.render_mesh(verts, faces, colors)

def render_mesh_dict(mesh, mode='npa',render_new=None):
    if render_new is None:
        # assert False
        image = renderer.render_mesh_dict(mesh, mode)[0]
    else:
        image = render_new.render_mesh_dict(mesh, mode)[0]

    image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image

def render_point_dict(points, render_new=None, color=True):
    if render_new is None:
        assert False
        # image = renderer.render_point_dict(points)
    else:
        image = render_new.render_point_dict(points, color=color)

    # image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image


def render_pytorch3d(mesh, mode='npa', renderer_new=None):
    if renderer_new is None:
        assert False
        # image = renderer.render_mesh_pytorch(mesh, mode=mode)[0]
    else:
        image = renderer_new.render_mesh_pytorch(mesh, mode=mode)[0]
    
    # if mode == 't':
    #     mask = image[:, :, 3]==0.
    #     image[:, :, 3] *= 255
    #     image[mask] = torch.tensor([[255,255,255,0],]).float().cuda()
    #     image = image.data.cpu().numpy().astype(np.uint8)
    # else:
    image = (255*image).data.cpu().numpy().astype(np.uint8)

    return image

def render_pytorch3d_point(points, mode='npa', renderer_new=None, train=False):
    if renderer_new is None:
        assert False
        # image = renderer.render_point_pytorch(points, mode=mode)[0]
    else:
        image, visible = renderer_new.render_point_pytorch(points, mode=mode)
        image = image[0]

    if train:
        return image, visible
    else:
    # if mode == 't':
    #     mask = image[:, :, 3]==0.
    #     image[:, :, 3] *= 255
    #     image[mask] = torch.tensor([[255,255,255,0],]).float().cuda()
    #     image = image.data.cpu().numpy().astype(np.uint8)
    # else:
        image = (255*image).data.cpu().numpy().astype(np.uint8)

        return image, visible


def render_joint(smpl_jnts, bone_ids, image_size=256):
    marker_sz = 6
    line_wd = 2

    image = np.ones((image_size, image_size,4), dtype=np.uint8)*255 
    smpl_jnts[:,1] += 0.3
    smpl_jnts[:,1] = -smpl_jnts[:,1] 
    smpl_jnts = smpl_jnts[:,:2]*image_size/2 + image_size/2

    for b in bone_ids:
        if b[0]<0 : continue
        joint = smpl_jnts[b[0]]
        cv2.circle(image, joint.astype('int32'), color=(0,0,0,255), radius=marker_sz, thickness=-1)

        joint2 = smpl_jnts[b[1]]
        cv2.circle(image, joint2.astype('int32'), color=(0,0,0,255), radius=marker_sz, thickness=-1)

        cv2.line(image, joint2.astype('int32'), joint.astype('int32'), color=(0,0,0,255), thickness=int(line_wd))

    return image


def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

                        
    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]


    # if weights.shape[1]> 30:
    #     colors= np.concatenate([np.array(cmap.colors)]*3)[:33]
    if weights.shape[1]> 30:
        colors= np.concatenate([np.array(cmap.colors)]*5)[:55]

    verts_colors = weights[:,:,None] * colors

    verts_colors = verts_colors.sum(1)

    return verts_colors
