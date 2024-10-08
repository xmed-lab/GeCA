"""
GeCA
"""
import math
import random
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
from utils import (
    xy_meshgrid,
    vit_positional_encoding,
    nerf_positional_encoding,
    pair,
    checkpoint_sequential,
    LocalizeAttention,
    ExtractOverlappingPatches
)
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        self.embedding_table = nn.Sequential(
            torch.nn.Linear(num_classes,
                            hidden_size),
            torch.nn.Linear(hidden_size,
                            hidden_size),
        )

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand((labels.shape[0], labels.shape[1]), device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
            # TODO add another self.num_classes to encode classifier guidance?
        labels = torch.where(drop_ids, 0, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, shift, scale, **kwargs):
        return self.fn(modulate(self.norm(x), shift, scale), **kwargs)


class FeedForward(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = torch.nn.Softmax(dim=-1)

        self.mask_heads = None
        self.attn_map = None

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x, localize=None, h=None, w=None, **kwargs):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if localize is not None:
            q = rearrange(q, 'b h n d -> b h n 1 d')
            k = localize(k, h, w)  # b h n (attn_height attn_width) d
            v = localize(v, h, w)  # b h n (attn_height attn_width) d

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b h n 1 (attn_height attn_width)

        attn = self.attend(dots)  # b h n 1 (attn_height attn_width)

        if kwargs.get('mask', False):
            mask = kwargs['mask']
            assert len(mask) <= attn.shape[1], 'number of heads to mask must be <= number of heads'
            attn[:, mask] *= 0.0

        self.attn_maps = attn

        out = torch.matmul(attn, v)  # b h n 1 d
        out = rearrange(out, 'b h n 1 d -> b n (h d)') if localize else rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(dim, 6 * dim, bias=True)
        )

        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))
            ]))

    def encode(self, x, c, attn, ff, localize_attn_fn=None, h=None, w=None, **kwargs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = gate_msa.unsqueeze(1) * attn(x, shift_msa, scale_msa, localize=localize_attn_fn, h=h, w=w, **kwargs) + x
        x = gate_mlp.unsqueeze(1) * ff(x, shift_mlp, scale_mlp) + x
        return x

    def forward(self, x, c, localize_attn_fn=None, h=None, w=None, **kwargs):
        if self.training and len(self.layers) > 1:
            # gradient checkpointing to save memory but at the cost of re-computing forward pass during backward pass
            funcs = [lambda _x: self.encode(_x, c, attn, ff, localize_attn_fn, h, w, **kwargs) for attn, ff in
                     self.layers]
            x = torch.utils.checkpoint.checkpoint_sequential(funcs, segments=len(funcs), input=x)
        else:
            for attn, ff in self.layers:
                x = self.encode(x, c, attn, ff, localize_attn_fn, h, w, **kwargs)
        return x


class GeCA(torch.nn.Module):
    def __init__(self, *,
                 input_size,
                 patch_size=8,
                 overlapping_patches=False,
                 num_patches=256,
                 octaves=0,
                 depth=1,
                 heads=4,
                 mlp_dim=64,
                 dropout=0.,
                 cell_init='random',
                 in_channels=4,
                 step_n=0,
                 cell_out_chns=4,
                 cell_hidden_chns=9,
                 embed_cells=True,
                 embed_dim=128,
                 embed_dropout=0.,
                 localize_attn=True,
                 localized_attn_neighbourhood=[3, 3],
                 pe_method='vit_handcrafted',
                 nerf_pe_basis='raw_xy',
                 nerf_pe_max_freq=5,
                 num_classes=11,
                 class_dropout_prob=0.1,
                 device='cuda',
                 alive_masking=False,
                 alpha_living_threshold=0.1) \
            :
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)
        cell_out_chns = in_channels * 2
        assert cell_init == 'constant' or cell_init == 'random' or cell_init == 'random_all'
        self.cell_init = cell_init
        self.localize_attn = localize_attn
        self.localized_attn_neighbourhood = localized_attn_neighbourhood
        self.localize_attn_fn = LocalizeAttention(localized_attn_neighbourhood, device) if localize_attn else None
        self.embed_cells = embed_cells
        self.pe_method = pe_method
        self.nerf_pe_basis = nerf_pe_basis
        self.nerf_pe_max_freq = nerf_pe_max_freq
        self.step_n = step_n
        self.patch_height, self.patch_width = pair(patch_size)
        self.overlapping_patches = overlapping_patches
        self.alive_masking = alive_masking
        self.alpha_living_threshold = alpha_living_threshold
        if patch_size == 1:
            self.overlapping_patches = False
        self.extract_overlapping_patches = \
            ExtractOverlappingPatches((self.patch_height, self.patch_width), self.device) \
                if self.overlapping_patches else None

        assert octaves >= 0
        self.octaves = octaves

        # computing dimensions for layers
        if self.pe_method == 'nerf_handcrafted':
            if self.nerf_pe_basis == 'sin_cos' or self.nerf_pe_basis == 'sinc':
                mult = 2 * 2 * self.nerf_pe_max_freq
            elif self.nerf_pe_basis == 'raw_xy':
                mult = 2
            elif self.nerf_pe_basis == 'sin_cos_xy':
                mult = 2 * 2 * self.nerf_pe_max_freq + 2
            self.cell_pe_patch_dim = mult * self.patch_height * self.patch_width \
                if not self.overlapping_patches else mult
        else:
            self.cell_pe_patch_dim = 0
        self.cell_in_patch_dim = in_channels * self.patch_height * self.patch_width \
            if not self.overlapping_patches else in_channels
        self.cell_out_patch_dim = cell_out_chns * self.patch_height * self.patch_width \
            if not self.overlapping_patches else cell_out_chns
        self.cell_hidden_chns = cell_hidden_chns
        self.cell_update_dim = self.cell_out_patch_dim + self.cell_hidden_chns
        self.cell_dim = \
            self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim + self.cell_hidden_chns \
                if not self.overlapping_patches else \
                self.cell_pe_patch_dim + (
                        in_channels * self.patch_height * self.patch_width) + self.cell_out_patch_dim + self.cell_hidden_chns
        if not embed_cells:
            embed_dim = self.cell_dim

        # rearranging from 2D grid to 1D sequence
        self.rearrange_cells = Rearrange('b c h w -> b (h w) c')

        if not self.overlapping_patches:
            self.patchify = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
                                      p1=self.patch_height, p2=self.patch_width)
            self.unpatchify = Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                                        p1=self.patch_height, p2=self.patch_width)
        else:
            self.patchify = torch.nn.Identity()
            self.unpatchify = torch.nn.Identity()

        self.cell_to_embedding = torch.nn.Linear(self.cell_dim, embed_dim) if embed_cells else None

        if pe_method == 'learned':
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.dropout = torch.nn.Dropout(embed_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)

        self.final_layer = FinalLayer(embed_dim, 1, self.cell_update_dim)

        self.in_channels = in_channels
        # don't update cells before first backward pass or else cell grid will have immensely diverged and grads will
        # be large and unhelpful
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        nn.init.constant_(self.transformer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.transformer.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def preprocess(self, cells, fn):
        pe_and_rgb_img = self.get_pe_and_rgb(cells)
        feats = fn(pe_and_rgb_img)
        feats_patch = self.patchify(feats)  # SLOW
        hidden = self.get_hidden(cells)
        preprocessed_cells = torch.concat([feats_patch, hidden], 1)
        return preprocessed_cells

    def f(self, cells, c, update_rate=0.5, **kwargs):
        _cells = cells
        if self.alive_masking:
            pre_life_mask = self.get_living_mask(cells[:, self.cell_pe_patch_dim + self.cell_in_patch_dim:])

        if self.overlapping_patches:
            neighbouring_inputs = self.extract_overlapping_patches(self.get_rgb_in(cells))
            _cells = torch.cat([self.get_pe_in(cells),
                                neighbouring_inputs,
                                self.get_rgb_out(cells),
                                self.get_hidden(cells)], 1)

        x = self.rearrange_cells(_cells)

        if self.embed_cells:
            x = self.cell_to_embedding(x)

        if self.pe_method == 'vit_handcrafted':
            x = x + vit_positional_encoding(x.shape[-2], x.shape[-1], device=self.device)
        elif self.pe_method == 'learned':
            x = x + self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x, c, localize_attn_fn=self.localize_attn_fn, h=cells.shape[-2], w=cells.shape[-1],
                             **kwargs)

        # stochastic cell state update
        b, _, h, w = cells.shape
        update = rearrange(self.final_layer(x, c), 'b (h w) c -> b c h w', h=h, w=w)
        if update_rate < 1.0:
            update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()
            updated = cells[:, self.cell_pe_patch_dim + self.cell_in_patch_dim:] + update_mask * update
        else:
            updated = cells[:, self.cell_pe_patch_dim + self.cell_in_patch_dim:] + update

        if self.alive_masking:
            post_life_mask = self.get_living_mask(updated)
            life_mask = (pre_life_mask & post_life_mask).float()
            updated[:, self.cell_out_patch_dim:] = updated[:, self.cell_out_patch_dim:] * life_mask

        cells = torch.cat([cells[:, :self.cell_pe_patch_dim + self.cell_in_patch_dim], updated], 1)

        return cells

    def get_living_mask(self, updated):
        max_pooled = torch.nn.functional.max_pool2d(
            updated[:, self.cell_out_patch_dim: self.cell_out_patch_dim + 1, :, :],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return (
                max_pooled > 0.1
        )

    def get_hidden_and_out(self, x):
        hidden = x[:, self.cell_pe_patch_dim + self.cell_in_patch_dim:]
        return hidden

    def forward(self, rgb_in, t, y, multi_scale=False, extras=None, step_n=1, update_rate=0.5, chkpt_segments=1,
                **kwargs):
        rgb_in_state = self.patchify(rgb_in)
        cells = torch.cat([rgb_in_state, extras], dim=1)

        # step_n = np.random.randint(8, 32)
        step_n = self.step_n
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c_ = t + y  # (N, D)

        inputs = []
        if self.octaves > 0:
            b, c, h, w = cells.shape
            octave = self.octaves
            while octave > 0 and h > 2 and w > 2:
                # let cells collect info before fusing
                cells = self.f(self.f(cells, update_rate, **kwargs), update_rate, **kwargs)
                # save input before fusing
                inputs.append(cells[:, :self.cell_pe_patch_dim + self.cell_in_patch_dim].detach().clone())
                cells = self.fusion(cells)  # fuse cells
                octave -= 1
                b, c, h, w = cells.shape

        if self.training and chkpt_segments > 1:
            # gradient checkpointing to save memory but at the cost of re-computing forward pass
            # during backward pass
            z_star = checkpoint_sequential(self.f, cells, c=c_, segments=chkpt_segments, seq_length=step_n,
                                           update_rate=update_rate, kwargs=kwargs)
        else:
            z_star = cells
            for _ in range(step_n):
                z_star = self.f(z_star, c_, update_rate, **kwargs)

        if self.octaves > 0:
            octave = self.octaves
            while octave > 0:
                z_star = self.mitosis(z_star)  # duplicate cells
                # replace input with input used at same scale before fusion
                z_star[:, :self.cell_pe_patch_dim + self.cell_in_patch_dim] = inputs.pop()
                # let cells adapt to the change
                z_star = self.f(self.f(z_star, update_rate, **kwargs), update_rate, **kwargs)
                octave -= 1

        if self.training:
            return z_star
        else:
            return self.get_rgb_out(z_star), self.get_hidden_and_out(z_star)

    def forward_with_cfg(self, x, t, y, cfg_scale, extras):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        # Guided Sampling
        n = combined.shape[0]
        size = (self.input_size // self.patch_height, self.input_size // self.patch_width)

        rgb_out_state = torch.zeros(n, self.cell_out_patch_dim, size[0], size[1],
                                    device=self.device)
        rgb_out_state[:, :, size[0] // 2, size[1] // 2] = torch.randn(n, self.cell_out_patch_dim,
                                                                                      device=self.device)
        extras[:, :self.cell_out_patch_dim] = rgb_out_state

        model_out, cells = self.forward(combined, t, y, extras=extras)

        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1), cells

    def mitosis(self, cells):
        return cells.repeat_interleave(2, -2).repeat_interleave(2, -1)

    def fusion(self, cells):
        return torch.nn.functional.avg_pool2d(cells, kernel_size=2, stride=2, padding=0)

    def seed(self, rgb_in, sz):
        patch_height, patch_width = (self.patch_height, self.patch_width) if not self.overlapping_patches else (1, 1)

        assert sz[0] % patch_height == 0 and sz[1] % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        size = (sz[0] // patch_height, sz[1] // patch_width)

        # for storing input from external source
        assert sz[0] == rgb_in.shape[2] and sz[1] == rgb_in.shape[3]
        n = rgb_in.shape[0]
        rgb_in_state = self.patchify(rgb_in)

        if self.cell_init == 'random':
            # randomly initialize cell output channels between [0,1)
            rgb_out_state = torch.zeros(n, self.cell_out_patch_dim, size[0], size[1], device=self.device)
            rgb_out_state[:, :, size[0] // 2, size[1] // 2] = torch.randn(n, self.cell_out_patch_dim,
                                                                          device=self.device)
            # initialize hidden channels with 0 for inter-cell communication
            hidden_state = torch.zeros(n, self.cell_hidden_chns, size[0], size[1], device=self.device)

        elif self.cell_init == 'random_all':
            rgb_out_state = torch.randn(n, self.cell_out_patch_dim, size[0], size[1], device=self.device)
            hidden_state = torch.randn(n, self.cell_hidden_chns, size[0], size[1], device=self.device)
        elif self.cell_init == 'constant':
            # initialize celloutput  channels with 0.5 (gray image)
            rgb_out_state = torch.zeros(n, self.cell_out_patch_dim, size[0], size[1], device=self.device) + 0.5

            # initialize hidden channels with 0 for inter-cell communication
            hidden_state = torch.zeros(n, self.cell_hidden_chns, size[0], size[1], device=self.device)

        if self.pe_method == 'nerf_handcrafted':
            xy = xy_meshgrid(sz[0], sz[1], -1, 1, -1, 1, n, device=self.device)
            pe = nerf_positional_encoding(xy, self.nerf_pe_max_freq, self.nerf_pe_basis, device=self.device)
            pe = self.patchify(pe)
            seed_state = torch.cat([pe, rgb_in_state, rgb_out_state, hidden_state], 1)
        else:
            seed_state = torch.cat([rgb_out_state, hidden_state], 1)

        return seed_state

    def get_pe_in(self, x):
        pe_patch = x[:, :self.cell_pe_patch_dim]
        pe = self.unpatchify(pe_patch)
        return pe

    def get_rgb_in(self, x):
        rgb_patch = x[:, self.cell_pe_patch_dim:self.cell_pe_patch_dim + self.cell_in_patch_dim]
        rgb = self.unpatchify(rgb_patch)
        return rgb

    def get_rgb_out(self, x):
        rgb_patch = x[:, self.cell_pe_patch_dim + self.cell_in_patch_dim:
                         self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim]
        rgb = self.unpatchify(rgb_patch)
        return rgb

    def get_rgb(self, x):
        rgb_patch = x[:,
                    self.cell_pe_patch_dim:self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim]
        rgb = self.unpatchify(rgb_patch)
        return rgb

    def get_pe_and_rgb(self, x):
        pe_and_rgb_patch = x[:, :self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim]
        pe_and_rgb = self.unpatchify(pe_and_rgb_patch)
        return pe_and_rgb

    def get_hidden(self, x):
        hidden = x[:, self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim:]
        return hidden


class CALoss(torch.nn.Module):
    def __init__(self, rec_factor=1e2, overflow_factor=1e2):
        super().__init__()
        self.rec_factor = rec_factor
        self.overflow_factor = overflow_factor
        self.lpips = None

    def forward(self, model, results):
        cells = results['output_cells']
        hidden = model.get_hidden(cells)
        output = results['output_img']
        target = results['ground_truth']['x']

        losses = {}

        # L1 loss for image reconstruction task
        losses['rec_loss'] = self.rec_factor * torch.nn.functional.l1_loss(output, target)

        # Overflow loss to prevent cell state overflow
        hidden_overflow_loss = (hidden - torch.clip(hidden, -5.0, 5.0)).abs().mean()
        rgb_overflow_loss = (output - torch.clip(output, -10, 10)).abs().mean()
        losses['overflow_loss'] = self.overflow_factor * (hidden_overflow_loss + rgb_overflow_loss)

        return losses
