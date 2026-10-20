"""LFM2.5-230M model architecture and checkpoint loading in pure PyTorch.

Requires torch + Python standard library; no Transformers or safetensors package.
The official architecture has 8 gated depthwise convolutions and 6 GQA layers,
SwiGLU, RMSNorm, rotary positions, and a weight-tied language-model head.
Tiny mode uses a small random model for offline correctness checks.
"""
import json
import struct

import torch
from torch import nn
import torch.nn.functional as F

MODEL_URL = 'https://huggingface.co/LiquidAI/LFM2.5-230M/resolve/main/model.safetensors'
# For SFT from base, alternatively use LiquidAI/LFM2.5-230M-Base.
LAYERS = ['conv','conv','full_attention','conv','full_attention','conv',
          'full_attention','conv','full_attention','conv','full_attention',
          'conv','full_attention','conv']


def cfg(tiny=False):
    """True config from checkpoint; tiny mode is for local correctness smoke tests only."""
    return dict(vocab=96 if tiny else 65536, dim=32 if tiny else 1024,
                ff=64 if tiny else 2560, heads=4 if tiny else 16,
                kv_heads=2 if tiny else 8,
                layers=LAYERS[:3] if tiny else LAYERS,
                conv_kernel=3, rope_theta=1000000.0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                      + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Checkpoint parameter names must match: w1, w2, w3."""
    def __init__(self, d, ff):
        super().__init__()
        self.w1, self.w3 = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotary(q, k, theta):
    """Apply split-half RoPE to [batch, heads, time, head_dim]."""
    t, hd = q.shape[-2], q.shape[-1]
    pos = torch.arange(t, device=q.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(0, hd, 2, device=q.device, dtype=torch.float32)/hd)
    a = pos[:, None] * inv[None, :]
    a = torch.cat([a,a], dim=-1)
    co, si = a.cos().to(q.dtype)[None,None,:,:], a.sin().to(q.dtype)[None,None,:,:]
    def rot(x):
        h = x.shape[-1] // 2
        return x * co + torch.cat([-x[...,h:], x[...,:h]], dim=-1) * si
    return rot(q), rot(k)


class Attention(nn.Module):
    def __init__(self,c):
        super().__init__()
        d,h,kh = c['dim'],c['heads'],c['kv_heads']
        self.h, self.kh, self.hd, self.theta = h,kh,d//h,c['rope_theta']
        self.q_proj = nn.Linear(d,d,bias=False)
        self.k_proj = nn.Linear(d,kh*self.hd,bias=False)
        self.v_proj = nn.Linear(d,kh*self.hd,bias=False)
        self.out_proj = nn.Linear(d,d,bias=False)
        self.q_layernorm = RMSNorm(self.hd)
        self.k_layernorm = RMSNorm(self.hd)

    def forward(self,x):
        b,t,_ = x.shape
        q = self.q_layernorm(self.q_proj(x).reshape(b,t,self.h,self.hd)).transpose(1,2)
        k = self.k_layernorm(self.k_proj(x).reshape(b,t,self.kh,self.hd)).transpose(1,2)
        v = self.v_proj(x).reshape(b,t,self.kh,self.hd).transpose(1,2)
        q,k = rotary(q,k,self.theta)
        k=k.repeat_interleave(self.h//self.kh,dim=1)
        v=v.repeat_interleave(self.h//self.kh,dim=1)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out_proj(out.transpose(1,2).reshape(b,t,-1))


class ShortConv(nn.Module):
    """LFM2 double-gated causal depthwise 1-D convolution."""
    def __init__(self,c):
        super().__init__()
        d,k = c['dim'],c['conv_kernel']
        self.in_proj=nn.Linear(d,3*d,bias=False)
        self.conv=nn.Conv1d(d,d,k,groups=d,bias=False,padding=k-1)
        self.out_proj=nn.Linear(d,d,bias=False)

    def forward(self,x):
        b,g,z=self.in_proj(x).transpose(1,2).chunk(3,dim=1)
        y=F.conv1d(b*z,self.conv.weight,bias=None,
                   padding=self.conv.kernel_size[0]-1,groups=b.shape[1])
        y=y[...,:x.size(1)]
        return self.out_proj((g*y).transpose(1,2))


class DecoderLayer(nn.Module):
    def __init__(self,c,layer_type):
        super().__init__()
        self.is_attn = layer_type == 'full_attention'
        if self.is_attn:
            self.self_attn=Attention(c)
        else:
            self.conv=ShortConv(c)
        self.feed_forward=SwiGLU(c['dim'],c['ff'])
        self.operator_norm=RMSNorm(c['dim'])
        self.ffn_norm=RMSNorm(c['dim'])

    def forward(self,x):
        x=x+(self.self_attn(self.operator_norm(x)) if self.is_attn
             else self.conv(self.operator_norm(x)))
        return x+self.feed_forward(self.ffn_norm(x))


class LFMBackbone(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.embed_tokens=nn.Embedding(c['vocab'],c['dim'],padding_idx=0)
        self.layers=nn.ModuleList([DecoderLayer(c,k) for k in c['layers']])
        self.embedding_norm=RMSNorm(c['dim'])

    def forward(self,ids):
        x=self.embed_tokens(ids)
        for layer in self.layers:
            x=layer(x)
        return self.embedding_norm(x)


class LFMPolicy(nn.Module):
    """LFM language model, optionally with a value head sharing its backbone."""
    def __init__(self,c,critic=False):
        super().__init__()
        self.model=LFMBackbone(c)
        self.lm_head=nn.Linear(c['dim'],c['vocab'],bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight  # checkpoint ties weights
        if critic:
            self.value_head=nn.Linear(c['dim'],1) # NEW critic, not in official checkpoint

    def forward(self,ids,with_values=False):
        hs=self.model(ids)
        logits=self.lm_head(hs)
        values=self.value_head(hs).squeeze(-1) if with_values else None
        return logits,values


class LFMCritic(nn.Module):
    """Independent LFM backbone + scalar value head; no language-model head."""
    def __init__(self,c):
        super().__init__()
        self.model=LFMBackbone(c)
        self.value_head=nn.Linear(c['dim'],1) # NEW head, not in official checkpoint

    def forward(self,ids):
        return self.value_head(self.model(ids)).squeeze(-1)


# Minimal safe(ish) safetensors reader using ONLY stdlib + torch.
# Only supports BF16/F16/F32/F64; no arbitrary pickled code execution.
DTYPES={'BF16':torch.bfloat16,'F16':torch.float16,
        'F32':torch.float32,'F64':torch.float64}

def read_safetensors(filename):
    """Reads a HF .safetensors checkpoint; large file, CPU RAM required."""
    out={}
    with open(filename,'rb') as f:
        head_length=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(head_length))
        data_start=8+head_length
        for name,meta in header.items():
            if name=='__metadata__':
                continue
            d=DTYPES.get(meta['dtype'])
            if d is None:
                raise ValueError(f'Unsupported dtype for {name}: {meta["dtype"]}')
            start,end=meta['data_offsets']
            f.seek(data_start+start)
            data=f.read(end-start)
            out[name]=torch.frombuffer(bytearray(data),dtype=d).reshape(meta['shape'])
    return out


def load_weights(model,ckpt):
    """Load HF backbone weights; an added scalar head stays newly initialized.

    The checkpoint LM head is optional for tied policies and unused by LFMCritic.
    All backbone tensor names and shapes must match.
    """
    state=read_safetensors(ckpt)
    missing,unexpected=model.load_state_dict(state,strict=False)
    unexpected=[n for n in unexpected if n!='lm_head.weight']
    bad=[n for n in missing if not n.startswith('value_head.') and n!='lm_head.weight']
    if bad or unexpected:
        raise RuntimeError(f'Checkpoint mismatch! Missing {bad[:8]}, unexpected {unexpected[:8]}.')
    print(f'Loaded {len(state)} checkpoint tensors from {ckpt}.')
    del state


def initialize(tiny=False,ckpt=None,critic=False,device=None):
    """Initialize the policy, optionally with a shared-backbone value head."""
    return _initialize(tiny,ckpt,device,LFMPolicy,critic=critic)


def initialize_critic(tiny=False,ckpt=None,device=None):
    """Load a separate pretrained backbone and add a trainable scalar value head."""
    return _initialize(tiny,ckpt,device,LFMCritic)


def _initialize(tiny,ckpt,device,model_class,**kwargs):
    """Real model requires --checkpoint. Tiny mode is a small random smoke test."""
    if not tiny and not ckpt:
        raise ValueError('Real LFM2.5 requires --checkpoint /path/to/model.safetensors')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=model_class(cfg(tiny),**kwargs)
    if tiny:
        # Faithful small-weight initialization for testing random models.
        # PyTorch Embedding defaults to N(0,1), creating extremely peaked
        # tied LM logits; pretrained HF LFM uses small initial weights.
        for module in model.modules():
            if isinstance(module,(nn.Linear,nn.Embedding,nn.Conv1d)):
                nn.init.normal_(module.weight,mean=0.,std=0.02)
                if getattr(module,'bias',None) is not None:
                    nn.init.zeros_(module.bias)
    if ckpt:
        load_weights(model,ckpt)
    if device=='cuda':
        # FP32 weights use more memory but keep AdamW updates stable.
        # For real GPU runs, BF16/autocast can be added separately.
        pass
    return model.to(device),device
