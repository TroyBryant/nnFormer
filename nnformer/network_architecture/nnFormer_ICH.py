from einops import rearrange, repeat
from copy import deepcopy
from nnformer.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


group=96
class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, S, H, W, C = x.shape

    x = x.view(B, S // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


class SwinTransformerBlock_kv(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0, 0, 0]

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
            dim, input_resolution=input_resolution, window_size=self.window_size, num_heads=num_heads, out_dim=dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.PCM = nn.Sequential(
            nn.Conv3d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
            nn.BatchNorm3d(mlp_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(mlp_hidden_dim, dim, 3, 1, 1, 1, group),
            nn.BatchNorm3d(dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, dim, 3, 1, 1, 1, group),
        )

    def forward(self, x, mask_matrix, skip=None, x_up=None):

        assert self.shift_size == [0, 0, 0]
        #print(x.shape)
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        skip = skip.permute(0, 4, 1, 2, 3)
        x_up = x_up.permute(0, 4, 1, 2, 3)
        x = self.attn(skip, x_up)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, L, C)


        convX = self.drop_path(self.PCM(shortcut.view(B, S, H, W, C).permute(0, 4, 1, 2, 3).contiguous()).permute(0, 2, 3, 4, 1).contiguous().view(B, L, C))
        # FFN
        x = shortcut + self.drop_path(x) #+ convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WindowAttention_kv(nn.Module):

    def __init__(self, dim, input_resolution, window_size, num_heads, out_dim=None, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.out_dim = out_dim or dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.shift_size = 0


        # pad feature maps to multiples of window size
        self.pad_r = (self.window_size[2] - self.input_resolution[2] % self.window_size[2]) % self.window_size[2]
        self.pad_b = (self.window_size[1] - self.input_resolution[1] % self.window_size[1]) % self.window_size[1]
        self.pad_g = (self.window_size[0] - self.input_resolution[0] % self.window_size[0]) % self.window_size[0]

        self.sampling_offsets = nn.Sequential(
            nn.AvgPool3d(kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(),
            nn.Conv3d(dim, self.num_heads * 3, kernel_size=1, stride=1)
        )
        self.sampling_scales = nn.Sequential(
            nn.AvgPool3d(kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(),
            nn.Conv3d(dim, self.num_heads * 3, kernel_size=1, stride=1)
        )
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv3d(dim, out_dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(out_dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)



        trunc_normal_(self.relative_position_bias_table, std=.02)


        S, H, W = self.input_resolution
        S, H, W = S + self.shift_size + self.pad_g, H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r
        image_reference_s = torch.linspace(-1, 1, S)
        image_reference_h = torch.linspace(-1, 1, H)
        image_reference_w = torch.linspace(-1, 1, W)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h, image_reference_s),
                                      0).permute(0, 3, 2, 1).unsqueeze(0)
        window_reference = nn.functional.avg_pool3d(image_reference, kernel_size=self.window_size)
        window_num_s, window_num_h, window_num_w = window_reference.shape[-3:]

        window_reference = window_reference.reshape(1, 3, window_num_s, 1, window_num_h, 1, window_num_w, 1)
        # print(window_reference.shape)
        base_coords_s = torch.arange(self.window_size[0]) * 2 * self.window_size[0] / self.window_size[0] / (S - 1)
        base_coords_s = (base_coords_s - base_coords_s.mean())
        base_coords_h = torch.arange(self.window_size[1]) * 2 * self.window_size[1] / self.window_size[1] / (H - 1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.window_size[2]) * 2 * self.window_size[2] / self.window_size[2] / (W - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_s = base_coords_s.unsqueeze(dim=0).repeat(window_num_s, 1)
        assert expanded_base_coords_s.shape[0] == window_num_s
        assert expanded_base_coords_s.shape[1] == self.window_size[0]
        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.window_size[1]
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.window_size[2]
        expanded_base_coords_s = expanded_base_coords_s.reshape(-1)
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h, expanded_base_coords_s),
                             0).permute(0, 3, 2, 1).reshape(1, 3, window_num_s, self.window_size[0], window_num_h,
                                                            self.window_size[1], window_num_w, self.window_size[2])
        self.base_coords = (window_reference + coords).cuda()
        self.coords = coords.cuda()

    def forward(self, x, x_up, mask=None):

        B_, _, S, H, W = x.shape

        shortcut = x

        assert S == self.input_resolution[0]
        assert H == self.input_resolution[1]
        assert W == self.input_resolution[2]

        x = F.pad(x, (self.shift_size, self.pad_r, self.shift_size, self.pad_b, self.shift_size, self.pad_g))
        window_num_s, window_num_h, window_num_w = self.base_coords.shape[-6], self.base_coords.shape[-4], \
                                                   self.base_coords.shape[-2]
        coords = self.base_coords.repeat(B_ * self.num_heads, 1, 1, 1, 1, 1, 1, 1)
        sampling_offsets = self.sampling_offsets(x)
        num_predict_total = B_ * self.num_heads
        sampling_offsets = sampling_offsets.reshape(num_predict_total, 3, window_num_s, window_num_h, window_num_w)
        sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (W // self.window_size[2])
        sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (H // self.window_size[1])
        sampling_offsets[:, 2, ...] = sampling_offsets[:, 2, ...] / (S // self.window_size[0])

        sampling_scales = self.sampling_scales(x)
        sampling_scales = sampling_scales.reshape(num_predict_total, 3, window_num_s, window_num_h, window_num_w)

        coords = coords + self.coords * sampling_scales[:, :, :, None, :, None, :, None] + sampling_offsets[:, :, :,
                                                                                           None, :, None, :, None]
        sample_coords = coords.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(num_predict_total,
                                                                       self.window_size[0] * window_num_s,
                                                                       self.window_size[1] * window_num_h,
                                                                       self.window_size[2] * window_num_w, 3)

        qkv = self.qkv(shortcut).reshape(B_, 3, self.num_heads, self.out_dim // self.num_heads, S, H, W).transpose(1,
                                                                                                                   0).reshape(
            3 * B_ * self.num_heads, self.out_dim // self.num_heads, S, H, W)
        qkv = F.pad(qkv,
                    (self.shift_size, self.pad_r, self.shift_size, self.pad_b, self.shift_size, self.pad_g)).reshape(3,
                                                                                                                     B_ * self.num_heads,
                                                                                                                     self.out_dim // self.num_heads,
                                                                                                                     S + self.shift_size + self.pad_g,
                                                                                                                     H + self.shift_size + self.pad_b,
                                                                                                                     W + self.shift_size + self.pad_r)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        k_selected = F.grid_sample(
            k.reshape(num_predict_total, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                      H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(B_ * self.num_heads, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                  H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r)
        v_selected = F.grid_sample(
            v.reshape(num_predict_total, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                      H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(B_ * self.num_heads, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                  H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r)

        q = q.reshape(B_, self.num_heads, self.out_dim // self.num_heads, window_num_s, self.window_size[0],
                      window_num_h,
                      self.window_size[1], window_num_w,
                      self.window_size[2]).permute(0, 3, 5, 7, 1, 4, 6, 8, 2).reshape(
            B_ * window_num_s * window_num_h * window_num_w,
            self.num_heads,
            self.window_size[0] *
            self.window_size[1] *
            self.window_size[2],
            self.out_dim // self.num_heads)
        k = k_selected.reshape(B_, self.num_heads, self.out_dim // self.num_heads, window_num_s, self.window_size[0],
                               window_num_h, self.window_size[1], window_num_w,
                               self.window_size[2]).permute(0, 3, 5, 7, 1, 4, 6, 8, 2).reshape(
            B_ * window_num_s * window_num_h * window_num_w, self.num_heads,
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.out_dim // self.num_heads)
        v = v_selected.reshape(B_, self.num_heads, self.out_dim // self.num_heads, window_num_s, self.window_size[0],
                               window_num_h, self.window_size[1], window_num_w,
                               self.window_size[2]).permute(0, 3, 5, 7, 1, 4, 6, 8, 2).reshape(
            B_ * window_num_s * window_num_h * window_num_w, self.num_heads,
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.out_dim // self.num_heads)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, '(b ss hh ww) h (ws1 ws2 ws3) d -> b (h d) (ss ws1) (hh ws2) (ww ws3)', h=self.num_heads,
                        b=B_,
                        ss=window_num_s, hh=window_num_h, ww=window_num_w, ws1=self.window_size[0],
                        ws2=self.window_size[1], ws3=self.window_size[2])
        out = out[:, :, self.shift_size:S + self.shift_size, self.shift_size:H + self.shift_size,
              self.shift_size:W + self.shift_size]


        out = self.proj(out)
        out = self.proj_drop(out)
        # print(out)

        return out

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
        nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
        nn.init.constant_(self.sampling_scales[-1].weight, 0.)
        nn.init.constant_(self.sampling_scales[-1].bias, 0.)

class WindowAttention(nn.Module):

    def __init__(self, dim, input_resolution, window_size, num_heads, out_dim=None, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.out_dim = out_dim or dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.shift_size = 0


        # pad feature maps to multiples of window size
        self.pad_r = (self.window_size[2] - self.input_resolution[2] % self.window_size[2]) % self.window_size[2]
        self.pad_b = (self.window_size[1] - self.input_resolution[1] % self.window_size[1]) % self.window_size[1]
        self.pad_g = (self.window_size[0] - self.input_resolution[0] % self.window_size[0]) % self.window_size[0]

        self.sampling_offsets = nn.Sequential(
            nn.AvgPool3d(kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(),
            nn.Conv3d(dim, self.num_heads * 3, kernel_size=1, stride=1)
        )
        self.sampling_scales = nn.Sequential(
            nn.AvgPool3d(kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(),
            nn.Conv3d(dim, self.num_heads * 3, kernel_size=1, stride=1)
        )
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv3d(dim, out_dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(out_dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))


        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        S, H, W = self.input_resolution
        S, H, W = S + self.shift_size + self.pad_g, H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r
        image_reference_s = torch.linspace(-1, 1, S)
        image_reference_h = torch.linspace(-1, 1, H)
        image_reference_w = torch.linspace(-1, 1, W)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h, image_reference_s),
                                      0).permute(0, 3, 2, 1).unsqueeze(0)
        window_reference = nn.functional.avg_pool3d(image_reference, kernel_size=self.window_size)
        window_num_s, window_num_h, window_num_w = window_reference.shape[-3:]

        window_reference = window_reference.reshape(1, 3, window_num_s, 1, window_num_h, 1, window_num_w, 1)
        #print(window_reference.shape)
        base_coords_s = torch.arange(self.window_size[0]) * 2 * self.window_size[0] / self.window_size[0] / (S - 1)
        base_coords_s = (base_coords_s - base_coords_s.mean())
        base_coords_h = torch.arange(self.window_size[1]) * 2 * self.window_size[1] / self.window_size[1] / (H - 1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.window_size[2]) * 2 * self.window_size[2] / self.window_size[2] / (W - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_s = base_coords_s.unsqueeze(dim=0).repeat(window_num_s, 1)
        assert expanded_base_coords_s.shape[0] == window_num_s
        assert expanded_base_coords_s.shape[1] == self.window_size[0]
        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.window_size[1]
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.window_size[2]
        expanded_base_coords_s = expanded_base_coords_s.reshape(-1)
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h, expanded_base_coords_s),
                             0).permute(0, 3, 2, 1).reshape(1, 3, window_num_s, self.window_size[0], window_num_h,
                                                         self.window_size[1], window_num_w, self.window_size[2])
        self.base_coords = (window_reference + coords).cuda()
        self.coords = coords.cuda()

    def forward(self,x, mask=None, pos_embed=None):
        B_, _, S, H, W = x.shape



        shortcut = x

        assert S == self.input_resolution[0]
        assert H == self.input_resolution[1]
        assert W == self.input_resolution[2]

        x = F.pad(x, (self.shift_size, self.pad_r, self.shift_size, self.pad_b, self.shift_size, self.pad_g))
        window_num_s, window_num_h, window_num_w = self.base_coords.shape[-6], self.base_coords.shape[-4], \
                                                   self.base_coords.shape[-2]
        coords = self.base_coords.repeat(B_ * self.num_heads, 1, 1, 1, 1, 1, 1, 1)
        sampling_offsets = self.sampling_offsets(x)
        num_predict_total = B_ * self.num_heads
        sampling_offsets = sampling_offsets.reshape(num_predict_total, 3, window_num_s, window_num_h, window_num_w)
        sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (W // self.window_size[2])
        sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (H // self.window_size[1])
        sampling_offsets[:, 2, ...] = sampling_offsets[:, 2, ...] / (S // self.window_size[0])

        sampling_scales = self.sampling_scales(x)
        sampling_scales = sampling_scales.reshape(num_predict_total, 3, window_num_s, window_num_h, window_num_w)

        coords = coords + self.coords * sampling_scales[:, :, :, None, :, None, :, None] + sampling_offsets[:, :, :,
                                                                                           None, :, None, :, None]
        sample_coords = coords.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(num_predict_total, self.window_size[0] * window_num_s,
                                                                 self.window_size[1] * window_num_h,
                                                                 self.window_size[2] * window_num_w, 3)

        qkv = self.qkv(shortcut).reshape(B_, 3, self.num_heads, self.out_dim // self.num_heads, S, H, W).transpose(1,
                                                                                                              0).reshape(
            3 * B_ * self.num_heads, self.out_dim // self.num_heads, S, H, W)
        qkv = F.pad(qkv,
                    (self.shift_size, self.pad_r, self.shift_size, self.pad_b, self.shift_size, self.pad_g)).reshape(3,
                                                                                                                     B_ * self.num_heads,
                                                                                                                     self.out_dim // self.num_heads,
                                                                                                                     S + self.shift_size + self.pad_g,
                                                                                                                     H + self.shift_size + self.pad_b,
                                                                                                                     W + self.shift_size + self.pad_r)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        k_selected = F.grid_sample(
            k.reshape(num_predict_total, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                      H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(B_ * self.num_heads, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                  H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r)
        v_selected = F.grid_sample(
            v.reshape(num_predict_total, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                      H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(B_ * self.num_heads, self.out_dim // self.num_heads, S + self.shift_size + self.pad_g,
                  H + self.shift_size + self.pad_b, W + self.shift_size + self.pad_r)

        q = q.reshape(B_, self.num_heads, self.out_dim // self.num_heads, window_num_s, self.window_size[0], window_num_h,
                      self.window_size[1], window_num_w,
                      self.window_size[2]).permute(0, 3, 5, 7, 1, 4, 6, 8, 2).reshape(B_ * window_num_s * window_num_h * window_num_w,
                                                                                      self.num_heads,
                                                                                      self.window_size[0] *
                                                                                      self.window_size[1] *
                                                                                      self.window_size[2],
                                                                                      self.out_dim // self.num_heads)
        k = k_selected.reshape(B_, self.num_heads, self.out_dim // self.num_heads, window_num_s, self.window_size[0],
                               window_num_h, self.window_size[1], window_num_w,
                               self.window_size[2]).permute(0, 3, 5, 7, 1, 4, 6, 8, 2).reshape(
            B_ * window_num_s * window_num_h * window_num_w, self.num_heads,
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.out_dim // self.num_heads)
        v = v_selected.reshape(B_, self.num_heads, self.out_dim // self.num_heads, window_num_s, self.window_size[0],
                               window_num_h, self.window_size[1], window_num_w,
                               self.window_size[2]).permute(0, 3, 5, 7, 1, 4, 6, 8, 2).reshape(
            B_ * window_num_s * window_num_h * window_num_w, self.num_heads,
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.out_dim // self.num_heads)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, '(b ss hh ww) h (ws1 ws2 ws3) d -> b (h d) (ss ws1) (hh ws2) (ww ws3)', h=self.num_heads, b=B_,
                        ss=window_num_s, hh=window_num_h, ww=window_num_w, ws1=self.window_size[0],
                        ws2=self.window_size[1], ws3=self.window_size[2])
        out = out[:, :, self.shift_size:S + self.shift_size, self.shift_size:H + self.shift_size,
              self.shift_size:W + self.shift_size]

        out = self.proj(out)
        out = self.proj_drop(out)
        #print(out)

        return out


    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
        nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
        nn.init.constant_(self.sampling_scales[-1].weight, 0.)
        nn.init.constant_(self.sampling_scales[-1].bias, 0.)


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if tuple(self.input_resolution) == tuple(self.window_size):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = [0, 0, 0]

        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, window_size=to_3tuple(self.window_size), num_heads=num_heads, out_dim=dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.PCM = nn.Sequential(
            nn.Conv3d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
            nn.BatchNorm3d(mlp_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(mlp_hidden_dim, dim, 3, 1, 1, 1, group),
            nn.BatchNorm3d(dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, dim, 3, 1, 1, 1, group),
        )

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """

        B, L, C = x.shape
        S, H, W = self.input_resolution

        assert L == S * H * W, "input feature has wrong size"


        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.attn(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, L, C)

        convX = self.drop_path(
            self.PCM(shortcut.view(B, S, H, W, C).permute(0, 4, 1, 2, 3).contiguous()).permute(0, 2, 3, 4,
                                                                                               1).contiguous().view(
                B, L, C))
        # FFN
        x = shortcut + self.drop_path(x) + convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, tag=None):
        super().__init__()
        self.dim = dim
        if tag == 0:
            self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        elif tag == 1:
            self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])
        else:
            self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 1])

        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):

        B, L, C = x.shape

        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)

        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, 2 * C)
        return x


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, tag=None):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        if tag == 0:
            self.up = nn.ConvTranspose3d(dim, dim // 2, [1, 2, 2], [1, 2, 2])
        elif tag == 1:
            self.up = nn.ConvTranspose3d(dim, dim // 2, [2, 2, 2], [2, 2, 2])
        elif tag == 2:
            self.up = nn.ConvTranspose3d(dim, dim // 2, [2, 2, 2], [2, 2, 2], output_padding=[1, 0, 0])

    def forward(self, x, S, H, W):

        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, C // 2)

        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 i_layer=None
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth
        self.i_layer = i_layer
        # build blocks

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:

            if i_layer == 1:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=1)
            elif i_layer == 2:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=2)
            elif i_layer == 0:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=0)
            else:
                self.downsample = None
        else:
            self.downsample = None

    def forward(self, x, S, H, W):

        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            if self.i_layer != 1 and self.i_layer != 2:
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            else:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True,
                 i_layer=None
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(
            SwinTransformerBlock_kv(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
        )
        for i in range(depth - 1):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            )

        self.i_layer = i_layer
        if i_layer == 1:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=1)
        elif i_layer == 0:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=2)
        else:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=0)

    def forward(self, x, skip, S, H, W):

        x_up = self.Upsample(x, S, H, W)
        x = skip + x_up
        if self.i_layer == 1:
            S, H, W = S * 2, H * 2, W * 2
        elif self.i_layer == 0:
            S, H, W = (S * 2) + 1, H * 2, W * 2
        else:
            S, H, W = S, H * 2, W * 2
        attn_mask = None
        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up)
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask)

        return x, S, H, W


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1 = [1, patch_size[1] // 2, patch_size[2] // 2]
        stride2 = [1, patch_size[1] // 2, patch_size[2] // 2]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)  # B C Ws Wh Ww
        x = self.proj2(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x


class Encoder(nn.Module):

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 down_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3)
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // down_stride[i_layer][0], pretrain_img_size[1] // down_stride[i_layer][1],
                    pretrain_img_size[2] // down_stride[i_layer][2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""

        x = self.patch_embed(x)
        down = []

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()

                down.append(out)
        return down


class Decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2, 2, 2],
                 num_heads=[24, 12, 6],
                 window_size=4,
                 up_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // up_stride[i_layer][0], pretrain_img_size[1] // up_stride[i_layer][1],
                    pretrain_img_size[2] // up_stride[i_layer][2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding,
                i_layer=i_layer
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):

        outs = []
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2).contiguous()
            skips[index] = i
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]

            x, S, H, W, = layer(x, skips[i], S, H, W)
            out = x.view(-1, S, H, W, self.num_features[i])
            outs.append(out)
        return outs


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)

        return x


class nnFormer(SegmentationNetwork):

    def __init__(self, crop_size=[64, 128, 128],
                 embedding_dim=192,
                 input_channels=1,
                 num_classes=14,
                 conv_op=nn.Conv3d,
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                 patch_size=[2, 4, 4],
                 window_size=[4, 4, 8, 4],
                 down_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 deep_supervision=True):

        super(nnFormer, self).__init__()

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x: x)

        embed_dim = embedding_dim
        depths = depths
        num_heads = num_heads
        patch_size = patch_size
        window_size = window_size
        down_stride = down_stride
        self.model_down = Encoder(pretrain_img_size=crop_size, window_size=window_size, embed_dim=embed_dim,
                                  patch_size=patch_size, depths=depths, num_heads=num_heads, in_chans=input_channels,
                                  down_stride=down_stride)
        self.decoder = Decoder(pretrain_img_size=crop_size, embed_dim=embed_dim, window_size=window_size[::-1][1:],
                               patch_size=patch_size, num_heads=num_heads[::-1][1:], depths=depths[::-1][1:],
                               up_stride=down_stride[::-1][1:])

        self.final = []
        if self.do_ds:

            for i in range(len(depths) - 1):
                self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=patch_size))

        else:
            self.final.append(final_patch_expanding(embed_dim, num_classes, patch_size=patch_size))

        self.final = nn.ModuleList(self.final)

    def forward(self, x):

        seg_outputs = []
        skips = self.model_down(x)
        neck = skips[-1]

        out = self.decoder(neck, skips)

        if self.do_ds:
            for i in range(len(out)):
                seg_outputs.append(self.final[-(i + 1)](out[i]))

            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]






