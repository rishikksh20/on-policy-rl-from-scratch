# LFM3-1.7B: architecture and pure PyTorch implementation

This document describes **the exact config supplied for `LiquidAI/LFM3-1.7B-Preview`**, together with the three pasted official Hugging Face files: `configuration_lfm3.py`, `modeling_lfm3.py`, and the width-7 convolution inference kernels. Those supplied files are the architectural reference. The implementation in [lfm3.py](lfm3.py) follows this repository's [lfm.py](lfm.py) interfaces and requires only PyTorch and the Python standard library.

The model is a **dense causal hybrid decoder**, with **1,728,801,536 unique parameters** for this configuration. Every layer contains an operator followed by an MLP. The operators are **13 double-gated short convolutions, 6 full grouped-query attention layers, and 5 sliding grouped-query attention layers**. This is not a mixture-of-experts model.

## Exact checkpoint configuration

```json
{
  "architectures": ["Lfm3ForCausalLM"],
  "auto_map": {
    "AutoConfig": "configuration_lfm3.Lfm3Config",
    "AutoModel": "modeling_lfm3.Lfm3Model",
    "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM"
  },
  "bos_token_id": 124894,
  "conv_L_cache": 7,
  "conv_bias": false,
  "dtype": "bfloat16",
  "eos_token_id": 124900,
  "head_dim": 128,
  "hidden_size": 2048,
  "intermediate_size": 7680,
  "layer_types": [
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"
  ],
  "max_position_embeddings": 262144,
  "model_type": "lfm3",
  "norm_eps": 0.00001,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "num_key_value_heads": 4,
  "pad_token_id": 124893,
  "rope_parameters": {
    "full_attention": {
      "partial_rotary_factor": 0.5,
      "rope_theta": 1000000.0,
      "rope_type": "default"
    },
    "sliding_attention": {
      "rope_theta": 10000,
      "rope_type": "default"
    }
  },
  "sliding_window": 256,
  "tie_word_embeddings": true,
  "transformers_version": "5.12.1",
  "use_cache": true,
  "use_sandwich_norm": true,
  "vocab_size": 128000
}
```

The supplied checkpoint's **262,144** positional limit overrides the pasted configuration class's **128,000** default. `transformers_version`, `architectures`, and `auto_map` describe the original loading machinery; they do not require a Transformers dependency here. BF16 is the checkpoint dtype; the local initializer defaults to FP32, as `lfm.py` does, and accepts an explicit `dtype`.

## Layer order and dimensions

Indices below are zero-based, matching checkpoint paths such as `model.layers.2`.

| Operator | Layer indices | Count |
|---|---|---:|
| Short convolution | 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 | 13 |
| Full attention | 2, 6, 10, 14, 18, 22 | 6 |
| Sliding attention | 4, 8, 12, 16, 20 | 5 |

The stream width is 2,048; each MLP has intermediate width 7,680. Attention has 16 query heads, 4 key/value heads, and explicit head dimension 128. Each KV head serves four query heads. Convolution layers have no query/key/value projections or rotary positions. There are no learned position embeddings.

## Residual blocks and normalization

For each layer, with operator `O` and MLP `M`:

```text
u = x + post_operator_norm(O(operator_norm(x)))
y = u + post_ffn_norm(M(ffn_norm(u)))
```

All four norms have a learned width-2,048 scale, no bias, and epsilon `1e-5`. The two output norms are required by `use_sandwich_norm=true`; omitting them changes both the computation and checkpoint structure. If that option is disabled in a custom config, the output norms are omitted. After all 24 layers, `model.embedding_norm` applies one more RMSNorm before the output heads. Despite its name, this norm is at the **end** of the backbone, not immediately after token embedding.

RMSNorm computes the mean square and reciprocal square root in FP32, casts the normalized activations back to their input dtype, then multiplies by the learned scale. There is no mean subtraction. Residual additions use the ordinary PyTorch dtype behavior; the reference does not add special residual scaling.

Every MLP is bias-free SwiGLU:

```text
M(x) = w2(silu(w1(x)) * w3(x))
w1, w3: [7680, 2048]; w2: [2048, 7680]
```

The activation is SiLU on `w1` only. This is neither GELU nor a sigmoid-only gate.

## Double-gated causal short convolution

The normalized input is projected to three width-2,048 streams, in **`B, C, x` order**:

```text
B, C, z = in_proj(h).chunk(3, dim=-1)
s = B * z
y = out_proj(C * depthwise_causal_conv(s))
```

Both gates are raw linear outputs; neither uses sigmoid or SiLU. The depthwise convolution has weight shape `[2048, 1, 7]` and 2,048 groups. Each channel independently reads the current token and up to six earlier tokens. PyTorch Conv1d uses cross-correlation order: weight index 0 multiplies `s[t-6]`, and index 6 multiplies `s[t]`. Left zero-padding preserves causality; any extra right outputs are discarded. There is no activation inside the convolution.

`conv_L_cache=7` sets the **kernel width**, as well as the official cache's window length; it does not mean seven previous tokens plus the current token. With `conv_bias=false`, the convolution, its `in_proj`, and its `out_proj` all have no bias. For a custom `conv_bias=true` config, all three have biases.

Padding states are zeroed before `in_proj`, and `B*z` is masked again afterward so a projection bias cannot populate the convolution history. The final projection can still produce nonzero outputs at padding positions; padding is excluded from attention keys and must also be excluded from training losses.

## Gated grouped-query attention

For either attention type:

1. Project normalized hidden states with bias-free `q_proj`, `k_proj`, and `v_proj`.
2. Reshape to Q `[batch, 16, time, 128]` and K/V `[batch, 4, time, 128]`.
3. Apply separate learned per-head-dimension RMSNorms (`q_norm`, `k_norm`) to Q and K. Each norm has **128 parameters shared across heads**; V is not normalized.
4. Apply the layer type's RoPE to Q and K.
5. Compute causal scaled dot-product attention with scale `1/sqrt(128)` and zero dropout. KV heads are repeated four times for the local PyTorch SDPA call.
6. Multiply each head's output by `sigmoid(gate_proj(h))`, then concatenate heads and apply `o_proj`.

`gate_proj` maps 2,048 input features to **16 scalars**, one per query head per token, broadcast over 128 head features. It is computed from the same normalized input as Q/K/V. It acts **after** attention's value aggregation and **before** `o_proj`; it does not gate attention logits, Q, K, or V.

Full attention permits every earlier token in the same document. Sliding attention permits `0 <= query_index - key_index < 256`: **the current token plus at most 255 previous tokens**. This strict boundary follows the official [Transformers sliding-window mask](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py). The local implementation uses that same inequality. The window is based on sequence indices; explicit position IDs supply RoPE positions and document boundaries.

## RoPE: the partial-rotation detail that matters

Both attention types use split-half rotation, not adjacent-pair interleaving:

```text
rotate_half([a, b]) = [-b, a]       # a and b each have 64 features
rotated = x * cos(angle) + rotate_half(x) * sin(angle)
```

For full attention, let `i = 0..63`:

```text
inv_freq[i] = 1000000 ** (-2*i/128)   if i < 32
              0                      otherwise
angle = concat(position * inv_freq, position * inv_freq)
```

Only **32 frequency pairs / 64 total features** rotate, but the denominator remains **128**. In the actual split-half layout, rotated feature indices are **0–31 and 64–95**; untouched indices are **32–63 and 96–127**. Rotating a contiguous 64-feature prefix or using denominator 64 gives the wrong result. Zero frequencies make the untouched coordinates' cosine exactly 1 and sine exactly 0.

Sliding attention uses theta **10,000**, defaults to partial factor **1.0**, and rotates all 64 pairs / 128 features. The two attention types cannot share the same frequencies. No RoPE is applied to V or to convolution layers.

The official implementation builds frequencies and trigonometric functions in FP32, casts cosine/sine to activation dtype, then performs the rotation arithmetic on FP32 Q/K and casts back. The local `rotary` follows this order, including the intermediate cosine/sine cast for BF16 fidelity. The provided config uses `rope_type="default"` without dynamic scaling; the local implementation rejects other RoPE types rather than silently approximating them.

## Padding, position IDs, and packed training

The local backbone/policy/critic accept optional `attention_mask` and `position_ids`:

- `attention_mask`: a 2D `[batch, time]` keep mask (1/True = token, 0/False = padding). Padding is not inferred from token IDs. The embedding's padding index is 124,893, **not 0**.
- `position_ids`: `[1, time]` or `[batch, time]` nonnegative integer positions. Without them, positions are `0..time-1`, matching the reference forward. For left-padded batches, supply positions explicitly, for example `(mask.long().cumsum(-1)-1).clamp_min(0)`.
- Without a padding mask, any position step other than `+1` starts a new document. For `[0,1,2,0,1]`, attention is block-causal and convolutions never read across the boundary. Resetting only RoPE without segmenting the convolution would leak data across documents.
- An explicit all-ones mask can accompany packed positions; it is equivalent to no padding mask. Packing together with a mask containing actual padding is rejected, matching the relevant reference restriction. Repeated zero positions in left padding do not count as packing when padding is explicitly masked.

This local API deliberately accepts 2D padding masks only; the reference's prepared mask dictionaries and custom 4D masks are framework integration interfaces. When packing examples for next-token loss, also mask labels at document boundaries: isolating hidden states does not prevent an incorrectly shifted loss from predicting the next document's first token.

## Checkpoint names and parameter count

Tensor names preserve the official layout. Attention naming differs from LFM2.5's `out_proj`, `q_layernorm`, and `k_layernorm`:

| Component | Example state-dict name | Shape |
|---|---|---|
| Token embeddings | `model.embed_tokens.weight` | `[128000, 2048]` |
| Conv input | `model.layers.0.conv.in_proj.weight` | `[6144, 2048]` |
| Depthwise kernel | `model.layers.0.conv.conv.weight` | `[2048, 1, 7]` |
| Conv output | `model.layers.0.conv.out_proj.weight` | `[2048, 2048]` |
| Attention Q | `model.layers.2.self_attn.q_proj.weight` | `[2048, 2048]` |
| Attention K/V | `model.layers.2.self_attn.k_proj.weight` / `v_proj.weight` | `[512, 2048]` each |
| Attention output | `model.layers.2.self_attn.o_proj.weight` | `[2048, 2048]` |
| Q/K norms | `model.layers.2.self_attn.q_norm.weight` / `k_norm.weight` | `[128]` each |
| Attention gate | `model.layers.2.self_attn.gate_proj.weight` | `[16, 2048]` |
| MLP | `model.layers.0.feed_forward.w1.weight` / `w3.weight` / `w2.weight` | See SwiGLU above |
| Four layer norms | `model.layers.0.{operator_norm,ffn_norm,post_operator_norm,post_ffn_norm}.weight` | `[2048]` each |
| Final norm | `model.embedding_norm.weight` | `[2048]` |
| Tied LM head | `lm_head.weight` | Same Parameter as token embeddings |

RoPE frequencies are nonpersistent buffers in the reference and introduce no checkpoint tensors or learned parameters.

| Component | Unique parameters |
|---|---:|
| Shared embedding / LM head | 262,144,000 |
| 24 SwiGLU MLPs | 1,132,462,080 |
| 13 convolution operators | 218,290,176 |
| 11 attention operators, including Q/K norms and gates | 115,706,624 |
| Four RMSNorms per layer | 196,608 |
| Final RMSNorm | 2,048 |
| **Total** | **1,728,801,536** |

Per-layer formulas for width `D=2048`, MLP width `F=7680`, Q heads `H=16`, KV heads `K=4`, and head dimension `A=128` are: MLP `3*D*F`; convolution `4*D*D + 7*D`; attention `2*D*H*A + 2*D*K*A + 2*A + D*H`; layer norms `4*D`. A shared or independent scalar value head adds `D+1 = 2049` parameters. A tied LM head adds zero unique parameters. An independent critic has its own embedding/backbone and no LM head.

## Local API and loading

`lfm3.py` retains `lfm.py`'s small dictionary configuration and functions: `cfg`, `RMSNorm`, `SwiGLU`, `rotary`, `Attention`, `ShortConv`, `DecoderLayer`, `LFMBackbone`, `LFMPolicy`, `LFMCritic`, `read_safetensors`, `load_weights`, `initialize`, and `initialize_critic`. `cfg()` uses the supplied configuration, expressed with the repository's short keys (`dim`, `ff`, `vocab`, etc.). `cfg(tiny=True)` shrinks dimensions and uses five layers so **all three operator types** are exercised; it is a random offline test model, not a pretrained 1.7B model.

```python
import torch
import lfm3

policy, device = lfm3.initialize(tiny=True, critic=True, device="cpu")
ids = torch.tensor([[1, 22, 33, 10]], device=device)
logits, values = policy(ids, with_values=True)
# logits: [1, 4, 96]; values: [1, 4]
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
)
loss.backward()
```

The policy always returns `(logits, values_or_None)`; the independent critic returns scalar values `[batch, time]`. These are the original repository conventions, rather than Transformers output objects. Loss computation and sampling stay in caller code. Both heads use the final normalized backbone states. Requesting values requires `critic=True` at construction.

```python
# Local checkpoint file, index JSON, or directory containing either:
policy, device = lfm3.initialize(ckpt="checkpoint/", device="cpu")
critic, _ = lfm3.initialize_critic(ckpt="checkpoint/", device=device)

# Optional BF16 storage for inference or a separately designed training setup:
policy, device = lfm3.initialize(
    ckpt="checkpoint/", device="cpu", dtype=torch.bfloat16
)
```

The stdlib safetensors reader supports BF16/F16/F32/F64 without executing checkpoint code or importing `safetensors`, Transformers, or `huggingface_hub`. The loader accepts a single `.safetensors` file or a local `model.safetensors.index.json` plus its shards. It reads one shard at a time. All backbone names/shapes must match; missing tied `lm_head.weight` is allowed, and an official LM head is ignored for a value-only critic. A newly added value head stays initialized locally. Unsupported or incomplete checkpoints fail explicitly. Quantized checkpoints are not supported.

Initialization defaults to FP32 for consistency with `lfm.py`. A real model requires an explicit local checkpoint; nothing is downloaded automatically. Tiny mode initializes Linear/Embedding/Conv1d weights with standard deviation 0.02, zeroes biases and the padding embedding row, and leaves RMSNorm scales at 1. The dtype option constructs weights directly in the requested floating dtype.

## Official cache behavior and implementation boundaries

The supplied reference supports a hybrid cache: full-attention KV history grows with the sequence, sliding layers keep a bounded recent KV window, and convolution layers keep projected **`B*z`** history rather than raw hidden states. The width-7 inference kernel accepts either six previous convolution inputs or a seven-slot stored window, discarding the oldest slot as necessary. Prefill and single-token decode have different kernel paths, but implement the same causal operator. The Triton path is an optional CUDA inference optimization; its PyTorch fallback defines the same architecture.

In the reference, packed documents must be prefilled in one forward into an empty cache. Their cache cannot subsequently be continued because attention would span documents; packed generation is rejected before prefill. Static-cache masks also impose restrictions on packed prefill and position-only padding. Full and sliding masks must be sized from a layer of their own type rather than the leading convolution layer. The local full-prefix implementation needs no cache sizing logic.

**Like `lfm.py`, `lfm3.py` implements training and full-prefix forwards without an inference cache.** `use_cache=true` in the original config describes the official implementation, not an enabled local cache. Sampling can recompute each prefix. There is no local `generate`, tokenizer, chat template, Transformers remote-code loading, Triton dependency, static cache, attention-backend registry, tensor-parallel plan, or framework gradient-checkpointing wrapper. These are integration/optimization features, not additional learned components. Adding caching later must preserve both attention histories and the convolution state.

The six existing RL trainers still import `lfm.py` and retain LFM2.5-specific EOS/vocabulary assumptions. Importing `lfm3` in a new caller provides the same basic model interfaces, but switching a trainer to the real LFM3 also requires LFM3 tokenization, special-token IDs, and suitable data. LFM2.5 checkpoints and token IDs are not interchangeable with this model.

## Memory and verification

Weights alone occupy about **3.22 GiB in BF16** or **6.44 GiB in FP32**, excluding checkpoint-loading buffers, activations, gradients, optimizer states, and extra actor/critic/reference copies. The positional limit is not a promise that a full 262,144-token training forward fits on a particular device. Six full-attention layers still entail quadratic attention work.

The simple sliding mask here is a dense `[time, time]` Boolean tensor, so it does not provide the reference's optimized long-context local-attention performance. For example, one such mask at the configured positional limit would consume 64 GiB. Full attention without padding/packing uses SDPA's causal flag and avoids constructing that mask, but padded/packed full attention also needs an explicit mask. This implementation is intended for understandable training code and shorter sequences.

If a cache were added, the six full-attention layers alone would use 12,288 bytes of BF16 KV storage per token per batch row (two tensors × six layers × four KV heads × 128 dimensions × two bytes), or 3 GiB at the configured maximum context. This excludes sliding and convolution states.

Run `python tests_lfm3.py` for focused offline checks of the exact architecture count and tensor layout, partial RoPE, reference-style eager attention and convolution equations, sliding-window boundaries, sandwich residuals, causality, padding/packing isolation, gradients, tied heads, and synthetic single/sharded checkpoint loading. Run `python tests_smoke.py` for the existing LFM2.5/RL regression checks. These tests do not establish numerical parity with the real pretrained checkpoint or measure model quality; that requires the actual checkpoint and an official-runtime comparison.
