import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'attn_map'])

# helpers functions


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        out = self.fn(x, *args, **kwargs)
        if isinstance(out, tuple):  # case: (out, attn)
            return out[0] + x, out[1]
        else:  # normal case
            return out + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)
        return out, attn
    
    
    
class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    

class TemporalAttentionModule(nn.Module):

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_x = self.conv1(attn)
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u

class Projector(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, out_channels=128):
        super(Projector, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # No activation on the last layer
        x = x.unsqueeze(2)
        return x   
    
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layer=1):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.cur_states = [None for _ in range(n_layer)]
        self.n_layer = n_layer
        self.conv_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

        self.conv_cans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=self.hidden_dim,  # for candidate neural memory
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape, device):
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = torch.zeros((b, self.hidden_dim, h, w), device=device)

    def step_forward(self, input_tensor, index):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = self.cur_states[index]
        assert h_cur is not None
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates[index](combined)

        reset_gate, update_gate = torch.split(torch.sigmoid(combined_conv), self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_cans[index](combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        self.cur_states[index] = h_next
        return h_next
    
    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor


# model
class ContextNet(nn.Module):
    def __init__(
        self,
        dim,    # must be same as Unet
        dim_mults=(1, 2, 4, 8),     # must be same as Unet
        channels = 1,
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        
        self.init_conv = nn.Conv2d(channels, dim, 7, padding = 3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # dims = [channels, *map(lambda m: dim * m, dim_mults)]
        # in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) -1 )
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in),
                        ConvGRUCell(dim_in, dim_in, 3, n_layer=1),
                        Downsample(dim_in, dim_out) if not is_last else nn.Identity()
                    ]
                )
            )

    
    def init_state(self, shape, device):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape, device)
            
    def forward(self, x):
        x = self.init_conv(x.float())
        context = []
        for i, (resnet, conv, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = conv(x)
            context.append(x)
            x = downsample(x)
        return context
    
    def scan_ctx(self, frames):
        b, t, c, h, w = frames.shape
        state_shape = (b, c, h, w)
        self.init_state(state_shape, frames.device)
        local_ctx = None
        globla_ctx = None
        
        for i in range(t):
            globla_ctx = self.forward(frames[:,i])
            if i == 5:
                local_ctx = [h.clone() for h in globla_ctx]
        return globla_ctx, local_ctx
        
        
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        T_in,
        dim_mults=(1, 2, 4, 8),
        self_condition = True,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.T_in = T_in
        self.channels = T_in * 2
        self.self_condition = self_condition
        input_channels = self.channels

        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        self.dims = dims
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        
        
        self.frag_idx_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # print(str(in_out))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in * 3, dim_in, time_emb_dim = time_dim * 2),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_in, TemporalAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim * 2),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_out, TemporalAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))


        self.out_dim = T_in

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim * 2)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None, ctx=None, idx=None):
        
        # x: (b, t, c, h, w)
        # cond: (b, t, c, h, w)
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        if exists(cond):
            z = cond[1]
            cond = rearrange(cond[0], 'b t c h w -> b (t c) h w')
        
        x = torch.cat((cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        f_idx = self.frag_idx_mlp(idx)
        t = torch.cat((t, f_idx), dim = 1)

        h = []

        for idx, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(torch.cat((x, ctx[idx], z[idx]),dim=1), t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x, attn_map = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.out_dim)
        return x, attn_map

# gaussian diffusion trainer class

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class MultiModalityAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim * 2, dim, 1)

    def forward(self, radar_input, satellite_input):
        b, c, h, w = radar_input.shape

        radar_input = rearrange(radar_input, 'b c h w -> b c 1 (h w)')
        satellite_input = rearrange(satellite_input, 'b c h w -> b c 1 (h w)')
        x = torch.cat([radar_input, satellite_input], 2)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        radar_out = out[:, :, :int(out.shape[2]/2), :]
        satellite_out = out[:, :, int(out.shape[2]/2): , :]
        radar_head_norms = torch.norm(radar_out, dim=[-1, -2])
        satellite_head_norms = torch.norm(satellite_out, dim=[-1, -2])
        head_norms = torch.stack([radar_head_norms,satellite_head_norms], dim=2)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = 2, y = h*w)
        out = rearrange(out, 'b c n (x y) -> b (c n) x y', x = h, y = w)
        out = self.to_out(out)
        return out, attn, head_norms

class MultiModalityEncoder(nn.Module):
    def __init__(
        self,
        dim,
        T_in,
        input_channel,
        dim_mults=(80, 80, 160, 320, 640),
    ):
        super().__init__()
        self.dim = dim
        self.T_in = T_in
        self.input_channel = input_channel
        self.dim_mults = dim_mults
        self.head_convs_radar = nn.ModuleList()
        self.head_convs_satellite = nn.ModuleList()
        self.head_attns = nn.ModuleList()

        for i, mult in enumerate(dim_mults):
            dim_in = self.T_in if i == 0 else dim_mults[i - 1]
            dim_out = dim_mults[i]
            layer_radar = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1) if 0 < i < len(dim_mults) - 1 else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            layer_satellite = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1) if 0 < i < len(dim_mults) - 1 else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            self.head_convs_radar.append(layer_radar)
            self.head_convs_satellite.append(layer_satellite)

            if i <= 1:
                self.head_attns.append(nn.Conv2d(dim_out*2, dim_out, 3, padding=1))
            else:
                self.head_attns.append(MultiModalityAttention(dim_out))
    
    def forward(self, radar_x, satellite_x):
        assert radar_x.shape == satellite_x.shape

        radar_x = rearrange(radar_x, 'b t c h w -> b (t c) h w')
        satellite_x = rearrange(satellite_x, 'b t c h w -> b (t c) h w')
        z = []
        attn_maps = []
        head_norm_maps = []
        for i, (conv_radar, conv_satellite, attn) in enumerate(zip(self.head_convs_radar, self.head_convs_satellite, self.head_attns)):
            radar_x = torch.relu(conv_radar(radar_x))
            satellite_x = torch.relu(conv_satellite(satellite_x))

            if i <= 1:
                x = attn(torch.cat((radar_x, satellite_x),1))
            else:
                x, attn_map, head_norm = attn(radar_x, satellite_x)
                if i > 2:
                    attn_maps.append(attn_map)
                head_norm_maps.append(head_norm)
            z.append(x)

        h = None
        attn_maps = torch.stack(attn_maps, dim=1)
        head_norm_maps = torch.stack(head_norm_maps, dim=1)
        return z, h, attn_maps, head_norm_maps

class ModalityEncoder(nn.Module):
    def __init__(
        self,
        dim,
        T_in,
        input_channel,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        self.dim = dim
        self.T_in = T_in
        self.input_channel = input_channel
        self.dim_mults = dim_mults
        self.head_convs = nn.ModuleList()
        

        for i, mult in enumerate(dim_mults):
            dim_in = self.input_channel * self.T_in if i == 0 else dim_mults[i - 1]
            dim_out = dim_mults[i]
            layer = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1) if 0 < i < len(dim_mults) - 1 else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            self.head_convs.append(layer)

        self.fc = nn.Linear(dim_out * 16 * 16, dim_out)
        self.ln = nn.LayerNorm(dim_out)
    
    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        z = []
        for i, conv in enumerate(self.head_convs):
            x = torch.relu(conv(x))
            z.append(x)

        h = None
        _ , _, H, W = x.shape
        x = rearrange(x, 'b k h w -> b (h w k)')
        h = self.ln(self.fc(x))
        return z, h

class ModalityDecoder(nn.Module):
    def __init__(
        self,
        T_in,
        input_channel,
        dim_mults=(1, 2, 4, 8)
    ):
        super().__init__()
        self.T_in = T_in
        self.input_channel = input_channel

        self.fc = nn.Linear(dim_mults[-1], dim_mults[-1] * 16 * 16)

        self.head_deconvs = nn.ModuleList()
        self.dim_mults = dim_mults.copy()
        self.dim_mults.reverse()
        for i, mult in enumerate(self.dim_mults):
            dim_in = self.dim_mults[i]
            dim_out = self.dim_mults[i+1] if i != len(self.dim_mults)-1 else T_in * input_channel
            layer = nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1) if 0 < i < len(self.dim_mults) - 1 else nn.ConvTranspose2d(dim_in, dim_out, 3, padding=1)
            self.head_deconvs.append(layer)
        
    def forward(self, z, h):
        h = torch.relu(self.fc(h))
        h = rearrange(h, 'b (c h w) -> b c h w', h = 16, w = 16)

        x = z+h
        for i, deconv in enumerate(self.head_deconvs):
            x = torch.relu(deconv(x))

        x = rearrange(x, 'b (c t) h w -> b t c h w', t = self.T_in)
        return x


class MIDERLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        # small trainable scalars (learned weights)
        self.w_conv = torch.nn.Parameter(torch.tensor(0.5))
        self.w_cov = torch.nn.Parameter(torch.tensor(0.5))

    def compute_modality_gate_features(self, radar, satellite):
        """
        radar: Tensor [B, T, C_r, H, W]
        satellite: Tensor [B, T, C_s, H, W]
        returns: Tensor [B, 2]  (F_conv, F_cov)
        """
        # --- Convection metric ---
        max_reflectivity = torch.amax(radar, dim=[1, 2, 3, 4])          # [B]
        min_cloud_temp = torch.amin(satellite, dim=[1, 2, 3, 4])         # [B]
        F_conv = max_reflectivity + (-min_cloud_temp)                 # lower temp = stronger convection

        # --- Coverage metric ---
        radar_var = torch.var(radar, dim=[1, 2, 3, 4])                   # [B]
        sat_mean = torch.mean(satellite, dim=[1, 2, 3, 4])               # [B]
        F_cov = radar_var + sat_mean

        F_gate = torch.stack([F_conv, F_cov], dim=1)                  # [B, 2]
        return F_gate
        

    def forward(self, head_norms, hard_mask, radar, satellite):
        """
        head_norms: [B, num_layers, num_heads, num_modalities]
        hard_mask: [B] boolean, True if sample ∈ B_hard
        """

        F_gate = self.compute_modality_gate_features(radar, satellite)
        mider_loss = []

        for l in range(head_norms.shape[1]):
            head_norm = head_norms[:,l]

            head_means = head_norm.mean(axis=(0, 2))
            global_mean_norm = head_norm.mean().item()
            min_head_norm_value = torch.min(head_means).item()
            if min_head_norm_value < (0.75 * global_mean_norm):
                dorm_head_index = torch.argmin(head_means).item()
            else:
                dorm_head_index = -1

            conv_head_index = torch.argmax(head_means).item()
            exclude_indices = [conv_head_index, dorm_head_index]
            if conv_head_index == dorm_head_index:
                exclude_indices = [conv_head_index]
            exclude_tensor = torch.tensor(exclude_indices, device=head_norm.device)
            all_head_indices = torch.arange(head_norm.shape[1], device=head_norm.device)
            mask_to_exclude = torch.isin(all_head_indices, exclude_tensor)
            wid_head_index = all_head_indices[~mask_to_exclude]
            
            head_norm_conv = head_norm[:, conv_head_index: conv_head_index + 1, :]
            head_norm_wid = head_norm.index_select(dim=1, index=wid_head_index)
            head_norm_dorm = head_norm[:,dorm_head_index:dorm_head_index+1,:]

            # compute τ_x per sample
            tau_conv = self.w_conv * F_gate[:, 0] + head_norm_conv.sum(dim=(1,2)).mean()
            tau_wid = self.w_cov * F_gate[:, 1] + head_norm_wid.sum(dim=(1,2)).mean()

            # total attention energy (sum across heads)
            total_energy_conv = head_norm_conv.sum(dim=(1,2))  # [B]
            total_energy_wid = head_norm_wid.sum(dim=(1,2))  # [B]
            total_energy_dorm = head_norm_dorm.sum(dim=(1,2))  # [B]

            # energy shortfall penalty
            energy_gap_conv = F.relu(tau_conv - total_energy_conv)
            energy_gap_wid = F.relu(tau_wid - total_energy_wid)
            energy_gap_dorm = F.relu(head_norm.sum(dim=(1,2)) * 0.5 - total_energy_dorm) 

            # apply only to hard samples
            loss  = (
                    (energy_gap_conv + energy_gap_wid + energy_gap_dorm) * hard_mask.float()
                ).mean()
            mider_loss.append(loss)

        mider_loss = torch.stack(mider_loss).mean()
        return mider_loss

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        ctx_net,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.ctx_net = ctx_net

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.b = betas
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.mse_loss = nn.MSELoss()

        self.encoder_radar = ModalityEncoder(
            dim = self.channels,
            T_in = self.model.T_in,
            input_channel = 1,
            dim_mults = self.model.dims
        )

        self.decoder_radar = ModalityDecoder(
            T_in = self.model.T_in,
            input_channel = 1,
            dim_mults = self.model.dims
        )

        self.encoder_satellite = ModalityEncoder(
            dim = self.channels,
            T_in = self.model.T_in,
            input_channel = 1,
            dim_mults = self.model.dims
        )

        self.decoder_satellite = ModalityDecoder(
            T_in = self.model.T_in,
            input_channel = 1,
            dim_mults = self.model.dims
        )

        self.encoder_combined = MultiModalityEncoder(
            dim = self.channels,
            T_in = self.model.T_in,
            input_channel = 2,
            dim_mults = self.model.dims
        )

        self.miderloss = MIDERLoss()

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @property
    def device(self):
        return self.betas.device
    
    def load_backbone(self, backbone_net):
        self.backbone_net = backbone_net

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond=None, ctx=None, idx=None, clip_x_start = False, rederive_pred_noise = False):
        model_output, attn_map = self.model(x.float(), t, cond=cond, ctx=ctx, idx=idx)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start, attn_map)

    def p_mean_variance(self, x, t, cond=None, ctx=None, idx=None, clip_denoised = True):
        preds = self.model_predictions(x, t, cond=cond, ctx=cond, idx=cond,)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond=None, ctx=None, idx=None,):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond=cond, ctx=ctx, idx=idx, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps = False):
        batch, device = shape[0], cond.device if cond is not None else self.device

        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            frames_pred, _ = self.p_sample(frames_pred, t, cond=cond, ctx=ctx, idx=idx)
            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # ret = self.unnormalize(ret)
        return ret


    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps = False):
        batch, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        device = cond[0].device if cond is not None else self.device
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        attn_maps = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=True):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            
            pred_noise, x_start, attn_map = self.model_predictions(frames_pred, time_cond, cond=cond, ctx=ctx, idx=idx, clip_x_start = True, rederive_pred_noise = True)
            attn_maps.append(attn_map)

            if time_next < 0:
                frames_pred = x_start
                imgs.append(frames_pred)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(frames_pred)

            frames_pred = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # ret = self.unnormalize(ret)
        attn_maps = torch.stack((attn_maps),1)
        return ret, attn_maps

    @torch.no_grad()
    def sample(self, frames_in, T_out, return_all_timesteps = False):
        radar_frame_in = frames_in[0]
        satellite_frame_in = frames_in[1]

        z_comb, h_comb, attn_maps, head_norm_maps = self.encoder_combined(radar_frame_in,satellite_frame_in) 

        frames_in = radar_frame_in

        B, T_in, c, h, w = frames_in.shape
        device = self.device
        backbone_output, _ = self.backbone_net.predict(frames_in)

        
        frames_in = self.normalize(frames_in)
        backbone_output = self.normalize(backbone_output)

        global_ctx, local_ctx = self.ctx_net.scan_ctx(torch.cat((frames_in, backbone_output), dim=1))
        
        # frames_in = rearrange(frames_in, 'b t c h w -> b (t c) h w')
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        frames_pred = []
        ys = []
        
        pre_frag = frames_in
        pre_mu = None
        for frag_idx in tqdm(range(T_out // T_in), desc="sampling frags:", disable=True):
            
            mu = backbone_output[:, frag_idx * T_in : (frag_idx + 1) * T_in]

            # Two strategies for channel condition
            # cond = pre_frag
            cond = pre_frag - pre_mu if  pre_mu is not None else torch.zeros_like(pre_frag)

            y, _ = sample_fn(
                (B, T_in, c, h, w), 
                cond=(cond,z_comb), 
                ctx=global_ctx if frag_idx > 0 else local_ctx,
                idx=torch.full((B,), frag_idx, device = device, dtype = torch.long), 
                return_all_timesteps = return_all_timesteps
                )

            frag_pred = y + mu
            
            frames_pred.append(
                frag_pred  # if frag_idx > 0 else mu
                )
            ys.append(y)
            
            pre_frag = frag_pred
            pre_mu = mu
        
        frames_pred = self.unnormalize(torch.cat(frames_pred, dim=1))
        frames_pred = frames_pred.clamp(0,1)
        # ys = self.unnormalize(torch.cat(ys, dim=1))
        ys = torch.cat(ys, dim=1)
        
        backbone_output = self.unnormalize(backbone_output)
        return frames_pred, backbone_output, ys, attn_maps, head_norm_maps
    

    def predict(self, frames_in, **kwargs):
        T_out = default(kwargs.get('T_out'), 20)
        pred, mu, y, attns_maps, head_norm_maps = self.sample(frames_in=frames_in, T_out=T_out)
        return pred, attns_maps, head_norm_maps
    
    def forward(self, frames_in, frames_gt, **kwargs ):
        radar_frame_in = frames_in[0]
        satellite_frame_in = frames_in[1]

        radar_frame_gt = frames_gt[0]
        satellite_frame_gt = frames_gt[1]

        z_radar, h_radar = self.encoder_radar(radar_frame_in)
        z_satellite, h_satellite = self.encoder_satellite(satellite_frame_in)
        z_comb, h_comb, attn_maps, head_norm_maps = self.encoder_combined(radar_frame_in,satellite_frame_in)

        frames_in = radar_frame_in
        frames_gt = radar_frame_gt

        B, T_in, c, h, w = frames_in.shape
        device = self.device
        T_out = frames_gt.shape[1]

        backbone_output, backbone_loss = self.backbone_net.predict(frames_in, frames_gt=frames_gt, compute_loss=True)
        
        frames_in = self.normalize(frames_in)
        backbone_output = self.normalize(backbone_output)
        frames_gt = self.normalize(frames_gt)

        residual = frames_gt - backbone_output
        global_ctx, local_ctx = self.ctx_net.scan_ctx(torch.cat((frames_in, backbone_output), dim=1))

        pre_frag = frames_in
        pre_mu = None
        pred_ress = []
        diff_loss = 0.

        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        for frag_idx in range(T_out // T_in):
            mu = backbone_output[:, frag_idx * T_in : (frag_idx + 1) * T_in]
            res = residual[:, frag_idx * T_in : (frag_idx + 1) * T_in]

            cond = pre_frag - pre_mu if pre_mu is not None else torch.zeros_like(pre_frag)
            res_pred, noise_loss = self.p_losses(res, t, cond=(cond,z_comb), ctx=global_ctx if frag_idx > 0 else local_ctx,
                                                    idx=torch.full((B,), frag_idx, device=device, dtype=torch.long))
            diff_loss += noise_loss

            pre_frag = frames_gt[:, frag_idx * T_in : (frag_idx + 1) * T_in]
            pre_mu = mu
        diff_loss /= (T_out // T_in)

        reconstruct_radar = self.decoder_radar(z_comb[-1], h_radar)
        reconstruct_satellite = self.decoder_satellite(z_comb[-1], h_satellite)

        recon_radar_loss = self.mse_loss(radar_frame_in, reconstruct_radar)
        recon_satellite_loss = self.mse_loss(satellite_frame_in, reconstruct_satellite)

        z_comb = F.avg_pool2d(z_comb[-1], kernel_size=z_comb[-1].shape[2:]).squeeze(-1).squeeze(-1)
        temperature = 0.1
        def safe_similarity(a, b):
            a = F.normalize(a, dim=1, eps=eps)
            b = F.normalize(b, dim=0, eps=eps)
            return torch.exp(torch.sum(a * b, dim=1) / temperature)

        eps = 1e-10
        sim_pos = safe_similarity(z_comb, z_comb.mean(dim=0))  # positive pair
        sim_neg1 = safe_similarity(z_comb, h_radar.mean(dim=0))
        sim_neg2 = safe_similarity(z_comb, h_satellite.mean(dim=0))

        sim_pos = torch.clamp(sim_pos, min=eps)
        sim_denominator = torch.clamp(sim_pos + sim_neg1 + sim_neg2, min=eps)

        loss_contrastive = -torch.log(sim_pos / sim_denominator).mean()

        hard_mask = diff_loss > diff_loss.mean()
        mider_loss = self.miderloss(head_norm_maps, hard_mask, radar_frame_in, satellite_frame_in)
        return {"phydnet":backbone_loss, 
                "recon_satellite": recon_satellite_loss, 
                "recon_radar": recon_radar_loss, 
                "contrast": loss_contrastive,
                "pixel_ep": diff_loss.mean(),
                "mider_loss": mider_loss
                }
    
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None, ctx=None, idx=None, cond=None):
        b, _, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise) # Use q_sample here for updating

        model_out, attn_map = self.model(x, t, cond=cond, ctx=ctx, idx=idx)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)

        return model_out, loss


def get_model(
    img_channels=1,
    dim = 64,
    dim_mults = (1,2,4,8),
    T_in = 5, 
    T_out = 20,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    **kwargs
):
    
    unet = Unet(
        dim = dim,
        T_in=T_in,
        dim_mults = dim_mults
    )
    
    context_net = ContextNet(
        dim = dim,
        dim_mults=dim_mults,
        channels=img_channels,
    )
    
    diffusion = GaussianDiffusion(
        model = unet,
        ctx_net = context_net,
        timesteps = timesteps,           # number of steps
        sampling_timesteps = sampling_timesteps,    
    )
    
    return diffusion
