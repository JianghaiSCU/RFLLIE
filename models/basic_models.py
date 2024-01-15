import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class res_block(nn.Module):
    def __init__(self, channels):
        super(res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class feature_pyramid(nn.Module):
    def __init__(self, channels):
        super(feature_pyramid, self).__init__()

        self.block0 = res_block(channels)

        self.down0 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = res_block(channels * 2)

        self.down1 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block2 = res_block(channels * 4)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        pyramid = []

        level0 = self.block0(x)
        pyramid.append(level0)
        level1 = self.block1(self.down0(level0))
        pyramid.append(level1)
        level2 = self.block2(self.down1(level1))
        pyramid.append(level2)
        return pyramid


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        b, c, h, w = x.shape

        x_up = F.interpolate(x, size=(h * 2, w * 2), mode='bilinear', align_corners=False)

        out = self.relu(self.conv(x_up))
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Depth_block(nn.Module):
    def __init__(self, channels):
        super(Depth_block, self).__init__()

        sequence = []
        sequence += [
            Depth_conv(channels, channels),
            nn.LeakyReLU(),
            Depth_conv(channels, channels),
            nn.LeakyReLU(),
            Depth_conv(channels, channels)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x
        return out


class Dilated_Resblock(nn.Module):
    def __init__(self, channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class global_to_local(nn.Module):
    def __init__(self, channels, up_sampling=True):
        super(global_to_local, self).__init__()

        self.depth_block = Depth_block(channels)
        self.attention = Attention(channels, num_heads=8, bias=False)
        self.dilated_block = Dilated_Resblock(channels)

        self.up_sampling = upsampling(channels, channels // 2)

        self.up = up_sampling

    def forward(self, x):

        if self.up:
            out = self.up_sampling(self.dilated_block(self.attention(self.depth_block(x))))
        else:
            out = self.dilated_block(self.attention(self.depth_block(x)))

        return out


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.pyramid = feature_pyramid(out_channels)

        self.level0_module = global_to_local(out_channels * 4, True)
        self.level1_module = global_to_local(out_channels * 2, True)
        self.level2_module = global_to_local(out_channels, False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        residual = x
        out0 = self.conv1(self.relu(self.conv0(x)))
        pyramid_fea = self.pyramid(out0)

        level0, level1, level2 = pyramid_fea[2], pyramid_fea[1], pyramid_fea[0]
        level0_out = self.level0_module(level0)
        level1_out = self.level1_module(level0_out + level1)
        level2_out = self.level2_module(level1_out + level2)

        out1 = self.relu(self.conv2(level2_out))
        final_out = self.relu(self.conv3(out1)) + residual

        return final_out
