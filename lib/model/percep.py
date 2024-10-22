import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(nn.Module):
    """https://github.com/elliottwu/unsup3d/blob/master/unsup3d/networks.py"""
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        device=torch.device('cuda:0')
        # Load VGG16 feature detector.
        # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        # with dnnlib.util.open_url(url) as f:
        self.vgg16 = torch.jit.load('./dataset/vgg16.pt').eval().to(device)
        # self.slice = nn.Sequential()
        # vgg16 = torchvision.models.vgg16(pretrained=True).eval().features
        # for x in range(16):
        #     self.slice.add_module(str(x), vgg16[x])
        # self.slice = self.slice.to(device)
        # for param in self.parameters():
        #     param.requires_grad = False
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, target_images, synth_images):
        # Features for target image.
        target_images = (target_images + 1) * (255/2)
        # mean = self.mean.to(target_images.device)
        # std = self.std.to(synth_images.device)
        # target_images = (target_images + 1) / 2
        # target_images = (target_images - mean) / std
        if target_images.shape[2] > 512:
            rand_start = torch.randint(256, (1,))
            # target_faces = target_images[:,:,rand_start[0]+20:rand_start[0]+532,256:768]
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
            # target_images = torch.cat([target_faces, target_images], dim=0)
        
        target_features = self.vgg16(target_images, resize_images=False, return_lpips=True)
       
        
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        # synth_images = (synth_images + 1) / 2
        # synth_images = (synth_images - mean) / std
        if synth_images.shape[2] > 512:
            # synth_faces = synth_images[:,:,rand_start[0]+20:rand_start[0]+532,256:768]
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
            # synth_images = torch.cat([synth_faces, synth_images], dim=0)
        synth_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
        # synth_features = self.vgg16(synth_images)

        dist = (target_features - synth_features).square().sum()
        # dist = (target_features - synth_features).square().mean()
        return dist

