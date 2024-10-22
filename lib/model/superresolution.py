import math

import torch
from torch import nn
import torch.nn.functional as F

from lib.torch_utils.ops.native_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from lib.model.discriminator import EqualConv2d, EqualConvTranspose2d

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    """Blur layer.

    Applies a blur kernel to input image using finite impulse response filter. Blurring feature maps after
    convolutional upsampling or before convolutional downsampling helps produces models that are more robust to
    shifting inputs (https://richzhang.github.io/antialiased-cnns/). In the context of GANs, this can provide
    cleaner gradients, and therefore more stable training.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    pad: tuple, int
        A tuple of integers representing the number of rows/columns of padding to be added to the top/left and
        the bottom/right respectively.
    upsample_factor: int
        Upsample factor.

    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class Upsample(nn.Module):
    """Upsampling layer.

    Perform upsampling using a blur kernel.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    factor: int
        Upsampling factor.

    """

    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):
    """Downsampling layer.

    Perform downsampling using a blur kernel.

    Args:
    ----
    kernel: list, int
        A list of integers representing a blur kernel. For exmaple: [1, 3, 3, 1].
    factor: int
        Downsampling factor.

    """

    def __init__(self, kernel=[1, 3, 3, 1], factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out

class ConvLayer2d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        layers = []

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(
                EqualConvTranspose2d(
                    in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate
                )
            )
            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            layers.append(
                EqualConv2d(in_channel, out_channel, kernel_size, padding=0, stride=2, bias=bias and not activate)
            )

        if (not downsample) and (not upsample):
            padding = kernel_size // 2

            layers.append(
                EqualConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=1, bias=bias and not activate)
            )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ToRGB(nn.Module):
    """Output aggregation layer.
    In the original StyleGAN2 this layer aggregated RGB predictions across all resolutions, but it's been slightly
    adjusted here to work with outputs of any dimension.
    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    z_dim: int
        Latent code dimension.
    upsample: bool
        Upsample the aggregated outputs.
    """

    def __init__(self, in_channel, out_channel, upsample=True):
        super().__init__()

        if upsample:
            self.upsample = Upsample()
            self.up = True
        else:
            self.up = False

        self.conv = ConvLayer2d(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=1,
            activate=False,
            bias=True,
        )

    def forward(self, input, skip=None):
        out = self.conv(input)

        if skip is not None:
            if self.up:
                skip = self.upsample(skip)
            out = out + skip
        return out


class ConvResBlock2d(nn.Module):
    """2D convolutional residual block with equalized learning rate.
    Residual block composed of 3x3 convolutions and leaky ReLUs.
    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    upsample: bool
        Apply upsampling via strided convolution in the first conv.
    downsample: bool
        Apply downsampling via strided convolution in the second conv.
    """

    def __init__(self, in_channel, out_channel, img_channels=3, upsample=False, downsample=False):
        super().__init__()

        assert not (upsample and downsample), 'Cannot upsample and downsample simultaneously'
        # mid_ch = in_channel if downsample else out_channel

        self.conv1 = ConvLayer2d(in_channel, out_channel, upsample=upsample, kernel_size=3)
        self.conv2 = ConvLayer2d(out_channel, out_channel, downsample=downsample, kernel_size=3)

        if (in_channel != out_channel) or upsample or downsample:
            self.skip = ToRGB(
                out_channel,
                img_channels,
                upsample=upsample
            )

    def forward(self, input, img=None):
        out = self.conv1(input)
        out = self.conv2(out)
        img = self.skip(out, img)
        return out, img


class SuperresolutionHybrid4X(torch.nn.Module):
    def __init__(self, channels, img_channels):
        super().__init__()
        self.block0 = ConvResBlock2d(channels, 128, img_channels=img_channels, upsample=False)
        self.block1 = ConvResBlock2d(128, 64, img_channels=img_channels, upsample=True)

    def forward(self, rgb, x):
        x, rgb = self.block0(x, rgb)
        x, rgb = self.block1(x, rgb)
        return rgb