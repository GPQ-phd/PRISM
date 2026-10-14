"""Self-contained Vision-RWKV token mixers used by PRISM.

The implementation follows the Vision-RWKV Bi-WKV summation equation and
quad-directional token shift, while using PyTorch prefix scans instead of the
optional CUDA extension from the reference repository.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def q_shift(x, patch_resolution, shift_pixel=1, gamma=0.25):
    """Shift four channel groups right, left, down, and up on a 2D grid."""
    if x.ndim != 3:
        raise ValueError("q_shift expects a [B, T, C] tensor")
    height, width = map(int, patch_resolution)
    batch, tokens, channels = x.shape
    if height <= 0 or width <= 0 or tokens != height * width:
        raise ValueError("patch_resolution must be positive and satisfy T = H * W")
    if not 0.0 <= gamma <= 0.25:
        raise ValueError("gamma must be in [0, 0.25]")
    if shift_pixel < 0:
        raise ValueError("shift_pixel must be non-negative")
    if shift_pixel == 0:
        return x

    spatial = x.transpose(1, 2).reshape(batch, channels, height, width)
    shifted = torch.zeros_like(spatial)
    group = int(channels * gamma)
    offset = min(int(shift_pixel), max(height, width))

    if group:
        horizontal = min(offset, width)
        vertical = min(offset, height)
        if horizontal < width:
            shifted[:, :group, :, horizontal:] = spatial[:, :group, :, : width - horizontal]
            shifted[:, group : 2 * group, :, : width - horizontal] = spatial[:, group : 2 * group, :, horizontal:]
        if vertical < height:
            shifted[:, 2 * group : 3 * group, vertical:, :] = spatial[:, 2 * group : 3 * group, : height - vertical, :]
            shifted[:, 3 * group : 4 * group, : height - vertical, :] = spatial[:, 3 * group : 4 * group, vertical:, :]
    shifted[:, 4 * group :, :, :] = spatial[:, 4 * group :, :, :]
    return shifted.flatten(2).transpose(1, 2)


class BiWKV(nn.Module):
    """Linear-complexity bidirectional WKV from Vision-RWKV Equation 5."""

    def __init__(self, channels, chunk_size=128):
        super().__init__()
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.spatial_decay = nn.Parameter(torch.zeros(self.channels))
        self.spatial_first = nn.Parameter(torch.zeros(self.channels))

    def _exclusive_past(self, centered_keys, values, decay_per_token):
        batch, tokens, channels = centered_keys.shape
        running_den = centered_keys.new_zeros(batch, channels)
        running_num = centered_keys.new_zeros(batch, channels)
        den_parts = []
        num_parts = []
        for start in range(0, tokens, self.chunk_size):
            end = min(start + self.chunk_size, tokens)
            index = torch.arange(start, end, device=centered_keys.device, dtype=centered_keys.dtype).view(1, -1, 1)
            exponent = centered_keys[:, start:end] + index * decay_per_token
            base = torch.exp(exponent.clamp(-60.0, 60.0))
            weighted = base * values[:, start:end]
            cumulative_den = base.cumsum(dim=1) + running_den[:, None, :]
            cumulative_num = weighted.cumsum(dim=1) + running_num[:, None, :]
            exclusive_den = cumulative_den - base
            exclusive_num = cumulative_num - weighted
            factor = torch.exp((-(index - 1.0) * decay_per_token).clamp(-60.0, 60.0))
            den_parts.append(exclusive_den * factor)
            num_parts.append(exclusive_num * factor)
            running_den = cumulative_den[:, -1]
            running_num = cumulative_num[:, -1]
        return torch.cat(num_parts, dim=1), torch.cat(den_parts, dim=1)

    def forward(self, k, v):
        if k.shape != v.shape or k.ndim != 3 or k.shape[-1] != self.channels:
            raise ValueError("BiWKV expects matching [B, T, C] key/value tensors")
        input_dtype = v.dtype
        keys = k.float()
        values = v.float()
        tokens = keys.shape[1]
        if tokens == 0:
            return v

        decay = self.spatial_decay.float().view(1, 1, -1) / float(tokens)
        first = self.spatial_first.float().view(1, 1, -1)
        key_offset = keys.amax(dim=1, keepdim=True).detach()
        centered = keys - key_offset

        past_num, past_den = self._exclusive_past(centered, values, decay)
        future_num, future_den = self._exclusive_past(
            centered.flip(1), values.flip(1), decay
        )
        future_num = future_num.flip(1)
        future_den = future_den.flip(1)

        current = torch.exp((centered + first).clamp(-60.0, 60.0))
        numerator = past_num + future_num + current * values
        denominator = past_den + future_den + current
        return (numerator / denominator.clamp_min(1e-12)).to(input_dtype)


def _mix_initialization(channels, num_layers, layer_id):
    ratio_depth = layer_id / max(num_layers - 1, 1)
    ratio_remaining = 1.0 - layer_id / max(num_layers, 1)
    position = torch.arange(channels, dtype=torch.float32) / max(channels, 1)
    base = position.view(1, 1, channels)
    return ratio_depth, ratio_remaining, base


class SpatialMix(nn.Module):
    def __init__(self, channels, num_layers, layer_id, shift_pixel=1, gamma=0.25):
        super().__init__()
        self.shift_pixel = shift_pixel
        self.gamma = gamma
        ratio_depth, ratio_remaining, base = _mix_initialization(channels, num_layers, layer_id)
        self.mix_k = nn.Parameter(base.pow(ratio_remaining))
        self.mix_v = nn.Parameter(base.pow(ratio_remaining) + 0.3 * ratio_depth)
        self.mix_r = nn.Parameter(base.pow(0.5 * ratio_remaining))
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.biwkv = BiWKV(channels)

        coordinate = torch.arange(channels, dtype=torch.float32) / max(channels - 1, 1)
        decay = -5.0 + 8.0 * coordinate.pow(0.7 + 1.3 * ratio_depth)
        zigzag = torch.tensor([((i + 1) % 3 - 1) * 0.5 for i in range(channels)])
        with torch.no_grad():
            self.biwkv.spatial_decay.copy_(decay)
            self.biwkv.spatial_first.copy_(math.log(0.3) + zigzag)

    def forward(self, x, patch_resolution):
        shifted = q_shift(x, patch_resolution, self.shift_pixel, self.gamma)
        xk = x * self.mix_k + shifted * (1.0 - self.mix_k)
        xv = x * self.mix_v + shifted * (1.0 - self.mix_v)
        xr = x * self.mix_r + shifted * (1.0 - self.mix_r)
        mixed = self.biwkv(self.key(xk), self.value(xv))
        return self.output(torch.sigmoid(self.receptance(xr)) * mixed)


class ChannelMix(nn.Module):
    def __init__(self, channels, num_layers, layer_id, shift_pixel=1, gamma=0.25, hidden_rate=4):
        super().__init__()
        self.shift_pixel = shift_pixel
        self.gamma = gamma
        _, ratio_remaining, base = _mix_initialization(channels, num_layers, layer_id)
        self.mix_k = nn.Parameter(base.pow(ratio_remaining))
        self.mix_r = nn.Parameter(base.pow(ratio_remaining))
        hidden = int(hidden_rate * channels)
        self.key = nn.Linear(channels, hidden, bias=False)
        self.value = nn.Linear(hidden, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)

    def forward(self, x, patch_resolution):
        shifted = q_shift(x, patch_resolution, self.shift_pixel, self.gamma)
        xk = x * self.mix_k + shifted * (1.0 - self.mix_k)
        xr = x * self.mix_r + shifted * (1.0 - self.mix_r)
        values = self.value(F.relu(self.key(xk)).square())
        return torch.sigmoid(self.receptance(xr)) * values


class VRWKVBlock(nn.Module):
    """Vision-RWKV block with externally supplied scene-adaptive norms."""

    def __init__(self, channels, num_layers, layer_id, norm_factory, hidden_rate=4):
        super().__init__()
        self.norm1 = norm_factory(channels)
        self.norm2 = norm_factory(channels)
        self.spatial_mix = SpatialMix(channels, num_layers, layer_id)
        self.channel_mix = ChannelMix(channels, num_layers, layer_id,hidden_rate=hidden_rate)

    def forward(self, x, state, patch_resolution):
        x = x + self.spatial_mix(self.norm1(x, state), patch_resolution)
        x = x + self.channel_mix(self.norm2(x, state), patch_resolution)
        return x
