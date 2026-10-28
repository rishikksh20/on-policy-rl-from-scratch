"""LFM3-1.7B architecture and checkpoint loading in pure PyTorch.

Adapted from the supplied official LFM3 reference, using lfm.py's interfaces.
Requires only torch + Python stdlib. See lfm3.md for architecture nuances.
Full-prefix training/forwards only: no KV/conv cache or tokenizer dependency.
"""
import json
import math
from pathlib import Path
import struct

import torch
from torch import nn
import torch.nn.functional as F


LAYERS = ['conv', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
          'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention', 'conv',
          'sliding_attention', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
          'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention', 'conv']
ATTENTION_TYPES = ('full_attention', 'sliding_attention')


def cfg(tiny=False):
    """Supplied 1.7B config; tiny is random and exercises all three operators."""
    return dict(vocab=96 if tiny else 128000, dim=32 if tiny else 2048,
                ff=64 if tiny else 7680, heads=4 if tiny else 16,
                kv_heads=2 if tiny else 4, head_dim=8 if tiny else 128,
                layers=list(LAYERS[:5] if tiny else LAYERS),
                conv_kernel=7, conv_bias=False, norm_eps=1e-5,
                use_sandwich_norm=True, sliding_window=256,
                max_position_embeddings=262144, tie_word_embeddings=True,
                pad_token_id=0 if tiny else 124893,
                bos_token_id=1 if tiny else 124894,
                eos_token_id=7 if tiny else 124900,
                dtype='bfloat16', use_cache=True,
                rope_parameters={
                    'full_attention': dict(rope_type='default', rope_theta=1000000.0,
                                           partial_rotary_factor=0.5),
                    'sliding_attention': dict(rope_type='default', rope_theta=10000.0),
                })


def _validate_config(c):
    for name in ('dim', 'ff', 'vocab', 'heads', 'kv_heads', 'head_dim', 'conv_kernel'):
        if not isinstance(c[name], int) or c[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if c['heads'] % c['kv_heads'] or c['head_dim'] % 2:
        raise ValueError('heads must be divisible by kv_heads and head_dim must be even')
    if not c['layers'] or any(k not in (*ATTENTION_TYPES, 'conv') for k in c['layers']):
        raise ValueError('layers must contain conv, full_attention, or sliding_attention')
    if c['norm_eps'] <= 0:
        raise ValueError('norm_eps must be positive')
    if c['pad_token_id'] is not None and not 0 <= c['pad_token_id'] < c['vocab']:
        raise ValueError('pad_token_id must be within the vocabulary')
    if 'sliding_attention' in c['layers']:
        window = c['sliding_window']
        if not isinstance(window, int) or window <= 0:
            raise ValueError('sliding_window must be a positive integer')
    for kind in set(c['layers']) & set(ATTENTION_TYPES):
        if kind not in c['rope_parameters']:
            raise ValueError(f'Missing rope_parameters for {kind}')
        p = c['rope_parameters'][kind]
        if p.get('rope_type', 'default') != 'default':
            raise ValueError('Only the supplied default RoPE is implemented')
        if p['rope_theta'] <= 0 or not 0 <= p.get('partial_rotary_factor', 1.0) <= 1:
            raise ValueError('RoPE theta must be positive and partial factor within [0, 1]')


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                       + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Official checkpoint names: w1 (SiLU), w3 (gate partner), w2 (output)."""
    def __init__(self, d, ff):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotary(q, k, theta, partial_rotary_factor=1.0, position_ids=None):
    """Split-half RoPE on [B,H,T,A], preserving full-head frequency spacing.

    Partial rotation zeroes the tail of A/2 frequencies, then duplicates the
    frequency vector. For A=128 and factor=.5 the rotating coordinates are
    0:32 and 64:96, not a contiguous 64-feature prefix. Match the reference's
    FP32 trig -> activation-dtype cos/sin -> FP32 rotation -> input-dtype order.
    """
    t, hd = q.shape[-2:]
    if position_ids is None:
        position_ids = torch.arange(t, device=q.device).unsqueeze(0)
    angles = int(partial_rotary_factor * hd) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, device=q.device,
                                      dtype=torch.float32) / hd))
    inv = F.pad(inv, (0, hd // 2 - angles))
    a = position_ids.to(device=q.device, dtype=torch.float32)[..., None] * inv
    a = torch.cat([a, a], dim=-1)
    co = a.cos().to(q.dtype).float().unsqueeze(1)
    si = a.sin().to(q.dtype).float().unsqueeze(1)

    def rot(x):
        xf = x.float()
        half = torch.cat([-xf[..., hd // 2:], xf[..., :hd // 2]], dim=-1)
        return (xf * co + half * si).to(x.dtype)

    return rot(q), rot(k)


class Attention(nn.Module):
    def __init__(self, c, layer_type='full_attention'):
        super().__init__()
        d, self.h, self.kh, self.hd = c['dim'], c['heads'], c['kv_heads'], c['head_dim']
        self.rope = dict(c['rope_parameters'][layer_type])
        self.sliding_window = c['sliding_window'] if layer_type == 'sliding_attention' else None
        self.q_proj = nn.Linear(d, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.kh * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.kh * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, c['norm_eps'])
        self.k_norm = RMSNorm(self.hd, c['norm_eps'])
        self.gate_proj = nn.Linear(d, self.h, bias=False)

    def forward(self, x, position_ids=None, attention_mask=None):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, t, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(b, t, self.kh, self.hd)).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.kh, self.hd).transpose(1, 2)
        q, k = rotary(q, k, self.rope['rope_theta'],
                      self.rope.get('partial_rotary_factor', 1.0), position_ids)
        k = k.repeat_interleave(self.h // self.kh, dim=1)
        v = v.repeat_interleave(self.h // self.kh, dim=1)
        if attention_mask is None and self.sliding_window is not None and t > self.sliding_window:
            pos = torch.arange(t, device=x.device)
            distance = pos[:, None] - pos[None, :]
            attention_mask = (distance >= 0) & (distance < self.sliding_window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                            is_causal=attention_mask is None, dropout_p=0.0)
        out = out.transpose(1, 2)  # [B,T,H,A]; one sigmoid gate per query head
        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj((out * gate).reshape(b, t, self.h * self.hd))


class ShortConv(nn.Module):
    """Double raw gates around a causal depthwise convolution; no activation."""
    def __init__(self, c):
        super().__init__()
        d, k, bias = c['dim'], c['conv_kernel'], c['conv_bias']
        self.in_proj = nn.Linear(d, 3 * d, bias=bias)
        self.conv = nn.Conv1d(d, d, k, groups=d, bias=bias, padding=k - 1)
        self.out_proj = nn.Linear(d, d, bias=bias)

    def forward(self, x, attention_mask=None, seq_idx=None):
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)
        b, g, z = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
        s = b * z
        if attention_mask is not None:
            s = s * attention_mask[:, None, :].to(s.dtype)
        if seq_idx is None:
            y = self.conv(s.to(self.conv.weight.dtype))[..., :x.size(1)].to(s.dtype)
        else:
            # Each width-k window reads only inputs from its own packed document.
            k = self.conv.kernel_size[0]
            windows = F.pad(s, (k - 1, 0)).unfold(-1, k, 1)
            docs = F.pad(seq_idx, (k - 1, 0), value=-1).unfold(-1, k, 1)
            same = (docs == docs[..., -1:]).unsqueeze(1)
            y = torch.einsum('bdtk,dk->bdt', windows * same.to(s.dtype),
                             self.conv.weight.squeeze(1))
            if self.conv.bias is not None:
                y = y + self.conv.bias[None, :, None]
            y = y.to(s.dtype)
        return self.out_proj((g * y).transpose(1, 2))


class DecoderLayer(nn.Module):
    def __init__(self, c, layer_type):
        super().__init__()
        self.is_attn = layer_type in ATTENTION_TYPES
        self.use_sandwich_norm = c['use_sandwich_norm']
        if self.is_attn:
            self.self_attn = Attention(c, layer_type)
        else:
            self.conv = ShortConv(c)
        self.feed_forward = SwiGLU(c['dim'], c['ff'])
        self.operator_norm = RMSNorm(c['dim'], c['norm_eps'])
        self.ffn_norm = RMSNorm(c['dim'], c['norm_eps'])
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(c['dim'], c['norm_eps'])
            self.post_ffn_norm = RMSNorm(c['dim'], c['norm_eps'])

    def forward(self, x, position_ids=None, attention_mask=None, seq_idx=None):
        h = self.operator_norm(x)
        h = (self.self_attn(h, position_ids, attention_mask) if self.is_attn
             else self.conv(h, attention_mask, seq_idx))
        if self.use_sandwich_norm:
            h = self.post_operator_norm(h)
        x = x + h
        h = self.feed_forward(self.ffn_norm(x))
        if self.use_sandwich_norm:
            h = self.post_ffn_norm(h)
        return x + h


def _positions_and_padding(ids, attention_mask, position_ids):
    """Validate 2D padding/position inputs and derive packed document IDs."""
    batch, length = ids.shape
    if attention_mask is not None:
        if attention_mask.shape != ids.shape:
            raise ValueError('attention_mask must have the same [batch, time] shape as ids')
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('attention_mask must contain only 0/1 keep flags')
        attention_mask = attention_mask.to(device=ids.device, dtype=torch.bool)
    explicit = position_ids is not None
    if not explicit:
        position_ids = torch.arange(length, device=ids.device).unsqueeze(0)
    elif (position_ids.ndim != 2 or position_ids.shape[1] != length
          or position_ids.shape[0] not in (1, batch)
          or position_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError('position_ids must be integer [1, time] or [batch, time]')
    position_ids = position_ids.to(device=ids.device)
    if explicit and bool((position_ids < 0).any()):
        raise ValueError('position_ids must be nonnegative')
    seq_idx = None
    if explicit and length > 1:
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None
        restarts = torch.diff(position_ids, dim=-1) != 1
        if attention_mask is not None:
            if bool((restarts & attention_mask[:, 1:] & attention_mask[:, :-1]).any()):
                raise ValueError('Packed documents cannot be combined with a padding mask; '
                                 'use unpadded position_ids restarting at 0')
        elif bool(restarts.any()):
            seq_idx = F.pad(restarts.to(torch.int64), (1, 0)).cumsum(-1)
    return position_ids, attention_mask, seq_idx


def _attention_masks(c, length, device, padding_mask=None, seq_idx=None):
    """Build masks once per forward; SDPA Boolean True means the key is allowed."""
    kinds = set(c['layers']) & set(ATTENTION_TYPES)
    masks = {}
    if not kinds:
        return masks
    plain = padding_mask is None and seq_idx is None
    if plain and ('sliding_attention' not in kinds or length <= c['sliding_window']):
        return dict.fromkeys(kinds)  # SDPA's causal flag handles every layer
    pos = torch.arange(length, device=device)
    distance = pos[:, None] - pos[None, :]
    causal = (distance >= 0)[None, None, :, :]
    if padding_mask is not None:
        causal = causal & padding_mask[:, None, None, :]
    if seq_idx is not None:
        causal = causal & (seq_idx[:, :, None] == seq_idx[:, None, :])[:, None]
    if 'full_attention' in kinds:
        masks['full_attention'] = None if plain else causal
    if 'sliding_attention' in kinds:
        masks['sliding_attention'] = causal & (distance < c['sliding_window'])
    return masks


class LFMBackbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        _validate_config(c)
        self.c = dict(c, layers=list(c['layers']))
        self.embed_tokens = nn.Embedding(c['vocab'], c['dim'], padding_idx=c['pad_token_id'])
        self.layers = nn.ModuleList([DecoderLayer(c, kind) for kind in c['layers']])
        self.embedding_norm = RMSNorm(c['dim'], c['norm_eps'])

    def forward(self, ids, attention_mask=None, position_ids=None):
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError('ids must be a nonempty [batch, time] tensor')
        positions, padding, docs = _positions_and_padding(ids, attention_mask, position_ids)
        masks = _attention_masks(self.c, ids.shape[1], ids.device, padding, docs)
        x = self.embed_tokens(ids)
        for kind, layer in zip(self.c['layers'], self.layers):
            mask = padding if kind == 'conv' else masks[kind]
            x = layer(x, positions, mask, docs)
        return self.embedding_norm(x)


class LFMPolicy(nn.Module):
    """Language model with an optional scalar value head, matching lfm.py."""
    def __init__(self, c, critic=False):
        super().__init__()
        self.model = LFMBackbone(c)
        self.lm_head = nn.Linear(c['dim'], c['vocab'], bias=False)
        if c['tie_word_embeddings']:
            self.lm_head.weight = self.model.embed_tokens.weight
        if critic:
            self.value_head = nn.Linear(c['dim'], 1)  # new head, absent from official checkpoint

    def forward(self, ids, with_values=False, attention_mask=None, position_ids=None):
        if with_values and not hasattr(self, 'value_head'):
            raise ValueError('Construct LFMPolicy with critic=True to request values')
        hs = self.model(ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(hs)
        values = self.value_head(hs).squeeze(-1) if with_values else None
        return logits, values


class LFMCritic(nn.Module):
    """Independent backbone and newly initialized scalar head; no LM head."""
    def __init__(self, c):
        super().__init__()
        self.model = LFMBackbone(c)
        self.value_head = nn.Linear(c['dim'], 1)

    def forward(self, ids, attention_mask=None, position_ids=None):
        hs = self.model(ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.value_head(hs).squeeze(-1)


# Minimal safetensors reader: torch + stdlib, no pickle or third-party loader.
DTYPES = {'BF16': torch.bfloat16, 'F16': torch.float16,
          'F32': torch.float32, 'F64': torch.float64}


def read_safetensors(filename):
    """Read one local floating-point safetensors file into CPU memory."""
    out = {}
    with open(filename, 'rb') as f:
        size = f.seek(0, 2)
        f.seek(0)
        if size < 8:
            raise ValueError('Truncated safetensors file')
        head_length = struct.unpack('<Q', f.read(8))[0]
        if head_length > size - 8:
            raise ValueError('Safetensors header extends beyond the file')
        header = json.loads(f.read(head_length))
        data_start = 8 + head_length
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dtype = DTYPES.get(meta['dtype'])
            if dtype is None:
                raise ValueError(f'Unsupported dtype for {name}: {meta["dtype"]}')
            shape = meta['shape']
            if any(not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError(f'Invalid shape for {name}: {shape}')
            start, end = meta['data_offsets']
            count = math.prod(shape)
            if (not 0 <= start <= end <= size - data_start
                    or end - start != count * dtype.itemsize):
                raise ValueError(f'Invalid safetensors offsets or shape for {name}')
            f.seek(data_start + start)
            data = bytearray(f.read(end - start))
            out[name] = (torch.frombuffer(data, dtype=dtype).reshape(shape) if count
                         else torch.empty(shape, dtype=dtype))
    return out


def _checkpoint_files(ckpt):
    path = Path(ckpt)
    if path.is_dir():
        index = path / 'model.safetensors.index.json'
        path = index if index.is_file() else path / 'model.safetensors'
    if path.name.endswith('.index.json'):
        index = json.loads(path.read_text())
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Checkpoint index must contain a nonempty weight_map')
        files = []
        root = path.parent.resolve()
        for name in dict.fromkeys(weight_map.values()):
            shard = (root / name).resolve()
            if root not in shard.parents or shard.suffix != '.safetensors':
                raise ValueError(f'Checkpoint shard must be a local .safetensors file: {name}')
            files.append(shard)
        return files, {n: (root / shard).resolve() for n, shard in weight_map.items()}
    if path.suffix != '.safetensors':
        raise ValueError('Expected a .safetensors file, checkpoint directory, or .index.json')
    return [path], None


def load_weights(model, ckpt):
    """Load exact official backbone names/shapes; leave added value heads alone.

    Accept one safetensors file or a local sharded checkpoint index/directory.
    Read/copy one shard at a time. Only a redundant tied LM head may be absent.
    """
    files, weight_map = _checkpoint_files(ckpt)
    expected = model.state_dict()
    backbone = {n for n in expected if not n.startswith('value_head.')}
    tied = (hasattr(model, 'lm_head') and
            model.lm_head.weight is model.model.embed_tokens.weight)
    required = backbone - ({'lm_head.weight'} if tied else set())
    seen, unexpected = set(), set()
    for path in files:
        state = read_safetensors(path)
        for name, tensor in state.items():
            if name in seen:
                raise RuntimeError(f'Duplicate checkpoint tensor: {name}')
            seen.add(name)
            if weight_map is not None and weight_map.get(name) != path.resolve():
                raise RuntimeError(f'Checkpoint index disagrees with shard for {name}')
            if name == 'lm_head.weight' and not hasattr(model, 'lm_head'):
                continue
            if name not in backbone:
                unexpected.add(name)
                continue
            if tensor.shape != expected[name].shape:
                raise RuntimeError(f'Checkpoint shape mismatch for {name}: '
                                   f'{tuple(tensor.shape)} != {tuple(expected[name].shape)}')
        # A missing tied head must never overwrite the shared embedding with random weights.
        accepted = {n: t for n, t in state.items() if n in backbone}
        model.load_state_dict(accepted, strict=False)
        del accepted, state
        tensor = None  # release the final loop reference before reading the next shard
    missing = required - seen
    if weight_map is not None:
        missing |= set(weight_map) - seen
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch! Missing {sorted(missing)[:8]}, '
                           f'unexpected {sorted(unexpected)[:8]}.')
    print(f'Loaded {len(seen)} checkpoint tensors from {ckpt}.')


def initialize(tiny=False, ckpt=None, critic=False, device=None, dtype=None):
    """Initialize a policy, optionally with a shared-backbone value head."""
    return _initialize(tiny, ckpt, device, LFMPolicy, dtype=dtype, critic=critic)


def initialize_critic(tiny=False, ckpt=None, device=None, dtype=None):
    """Load an independent backbone and add a trainable scalar value head."""
    return _initialize(tiny, ckpt, device, LFMCritic, dtype=dtype)


def _initialize(tiny, ckpt, device, model_class, dtype=None, **kwargs):
    if not tiny and not ckpt:
        raise ValueError('Real LFM3-1.7B requires --checkpoint /path/to/checkpoint')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError('dtype must be a floating-point torch dtype')
    # Construct via meta so an explicit BF16 choice needs no full FP32 allocation.
    # to_empty creates the CPU storage; the official backbone is then loaded below.
    with torch.device('meta'):
        model = model_class(cfg(tiny), **kwargs).to(dtype=dtype)
    model.to_empty(device='cpu')
    for module in model.modules():
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # A real backbone is overwritten by its checkpoint; only tiny weights
            # and new value heads need initialization.
            if tiny or module is getattr(model, 'value_head', None):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)
    if hasattr(model, 'lm_head') and model.model.c['tie_word_embeddings']:
        # to_empty may replace shared Parameters, so restore the tie explicitly.
        model.lm_head.weight = model.model.embed_tokens.weight
    if tiny and model.model.embed_tokens.padding_idx is not None:
        with torch.no_grad():
            model.model.embed_tokens.weight[model.model.embed_tokens.padding_idx].zero_()
    if ckpt:
        load_weights(model, ckpt)
    return model.to(device), device
