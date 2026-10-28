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

**Like `lfm.py`, `lfm3.py` implements training and full-prefix forwards without an inference cache.** `use_cache=true` in the original config describes the official implementation, not an enabled local cache. Sampling can recompute each prefix. The model module has no `generate`, tokenizer, or chat-template dependency; the separate `test_lfm3.py` script provides those inference helpers. There is no Transformers remote-code loading, Triton dependency, static cache, attention-backend registry, tensor-parallel plan, or framework gradient-checkpointing wrapper. These are integration/optimization features, not additional learned components. Adding caching later must preserve both attention histories and the convolution state.

The six existing RL trainers still import `lfm.py` and retain LFM2.5-specific EOS/vocabulary assumptions. Importing `lfm3` in a new caller provides the same basic model interfaces, but switching a trainer to the real LFM3 also requires LFM3 tokenization, special-token IDs, and suitable data. LFM2.5 checkpoints and token IDs are not interchangeable with this model.

## Memory and verification

Weights alone occupy about **3.22 GiB in BF16** or **6.44 GiB in FP32**, excluding checkpoint-loading buffers, activations, gradients, optimizer states, and extra actor/critic/reference copies. The positional limit is not a promise that a full 262,144-token training forward fits on a particular device. Six full-attention layers still entail quadratic attention work.

The simple sliding mask here is a dense `[time, time]` Boolean tensor, so it does not provide the reference's optimized long-context local-attention performance. For example, one such mask at the configured positional limit would consume 64 GiB. Full attention without padding/packing uses SDPA's causal flag and avoids constructing that mask, but padded/packed full attention also needs an explicit mask. This implementation is intended for understandable training code and shorter sequences.

If a cache were added, the six full-attention layers alone would use 12,288 bytes of BF16 KV storage per token per batch row (two tensors × six layers × four KV heads × 128 dimensions × two bytes), or 3 GiB at the configured maximum context. This excludes sliding and convolution states.

Run `python test_lfm3.py --unit` for focused offline checks of the exact architecture count and tensor layout, partial RoPE, reference-style eager attention and convolution equations, sliding-window boundaries, sandwich residuals, causality, padding/packing isolation, gradients, tied heads, and synthetic single/sharded checkpoint loading. Run `python tests_smoke.py` for the existing LFM2.5/RL regression checks. These offline tests do not establish numerical parity with the real pretrained checkpoint or measure model quality; that requires the actual checkpoint and an official-runtime comparison.

Both suites passed locally on CPU with PyTorch 2.8.0. An additional comparison executed the computational classes extracted from the supplied official source, replacing framework decorators/dispatch only, and loaded matching small random weights by exact tensor name. Across normal, padded, and packed inputs, with convolution bias and sandwich norms each enabled/disabled, FP32 outputs differed by less than `1e-5`; gradients also matched within the comparison tolerances. BF16 forward/backward checks passed, with the expected larger rounding differences between PyTorch SDPA and the reference's eager softmax.

## Real checkpoint generation test

[`test_lfm3.py`](test_lfm3.py) combines the offline suite and the supplied `LiquidAI/LFM3-1.7B` pipeline example in one file. Its code is grouped into configuration, downloads, tokenization, generation, test helpers, test cases, and the entry point. Test classes group architecture, positions/masks, model gradients, checkpoints, generation, and runner behavior. The default command runs the quick checks first and proceeds to real generation only if they pass.

The hosted release's architecture was checked against the supplied config. The script imports no `transformers`, `huggingface_hub`, `tokenizers`, or `safetensors` package. It uses stdlib HTTP for downloads, a local ByteLevel BPE implementation for `tokenizer.json`, the independent `regex` package for the tokenizer's Unicode pretokenization pattern, and Jinja2's sandbox to render the downloaded official chat template. `lfm3.py` and the offline test mode require only PyTorch and stdlib.

```bash
pip install torch regex jinja2

# Quick offline tests only (no tokenizer dependencies or checkpoint needed):
python test_lfm3.py --unit

# Run checks, then download, verify, load, and generate:
python test_lfm3.py

# Same run without any network access after the files have been downloaded:
python test_lfm3.py --offline

# Shorter run; thinking may consume the whole limit before an answer appears:
python test_lfm3.py --offline --max-new-tokens 64

# Optional different prompt:
python test_lfm3.py --offline --prompt "Explain grouped-query attention."
```

The CLI has four options: `--unit`, `--offline`, `--prompt`, and `--max-new-tokens`. Advanced settings live in the `SAMPLING` and `SPECULATION` dictionaries at the top of the file, alongside `MODEL_ID`, `DEFAULT_DIRECTORY`, and `PROMPT`. The device and storage dtype are automatic: CUDA with BF16 when CUDA is available, otherwise CPU with FP32. Results always go into the checkpoint directory. The normal generation path retains the official thinking-enabled template.

The defaults are `do_sample=True`, temperature `0.1`, top-k `50`, repetition penalty `1.1`, maximum **2,048 new tokens**, and local random seed `42`. The penalty is applied once per distinct token in the entire prompt/generated history, multiplying negative logits and dividing positive logits, followed by temperature scaling, top-k filtering, and multinomial sampling. Generation stops on the downloaded generation config's EOS IDs. A sampled answer need not be identical to a separate pipeline run, particularly when seeds, attention backends, dtypes, or caching differ.

### Live streaming and speculative decoding

Every committed token is sent immediately to an incremental UTF-8 decoder, and each complete text fragment is flushed to standard output. Byte fragments of a Unicode character remain buffered until the character is complete. The live output includes the default `<think>` reasoning section followed by the answer. The final answer is not printed a second time. The saved JSON still separates reasoning from assistant content.

Speculative decoding is enabled by default using [prompt lookup](https://github.com/apoorvumang/prompt-lookup-decoding): drafts come from repeated token spans in the context, so no second model or checkpoint is needed. In this implementation, a repeated suffix of two to four tokens proposes up to four continuation tokens. The target evaluates the prefix plus the draft in a single causal forward and computes only the needed verification/bonus LM-head rows.

Verification draws from the target's full configured sampling distribution at each position. A proposed token is accepted when that draw matches it; otherwise the target draw is committed, and later draft tokens and their logits are discarded. With a deterministic proposal, acceptance probability is exactly the target probability of that proposed token, while a mismatch already follows the target distribution conditioned on rejection. This specializes the distribution-preserving verification principle of [speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html) to a point-mass draft. Penalties use only the committed prefix, including previously accepted tokens in the current block. Drafts are never streamed before verification.

When every proposed token matches, the same forward supplies an additional target-sampled token. EOS and the token/context limits still stop generation immediately. With no repeated suffix, the loop makes an ordinary one-token step. Because the backbone currently has no inference cache, each verification round recomputes the prefix; there are no stale KV or convolution states to roll back. Throughput gains depend on repetition and draft acceptance; enabling speculation does not guarantee a speedup for every prompt. Set `SPECULATION['enabled'] = False` in the configuration block to compare ordinary decoding.

The end summary reports `Speed: ... tok/s`, generated-token count, elapsed time, accepted/proposed draft tokens, and target forward passes. CUDA is synchronized before and after timing. The measurement includes the first prompt forward, sampling, verification, and streaming; it excludes downloads, model loading, and offline tests. Token counts include EOS when emitted. The JSON `performance` object also records time to first token and draft acceptance rate.

The official template defaults to thinking enabled, so the exact initial prompt is:

```text
<|startoftext|><|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
<think>
```

For the tested revision, its 17 token IDs are `[124894, 124899, 5922, 207, 2992, 355, 340, 22, 13729, 497, 39, 124900, 207, 124899, 63514, 207, 124901]`. The tokenizer implements the checkpoint's `ignore_merges=true` behavior, added tokens, byte alphabet, merge ranks, and UTF-8 decoding. Its encoding/decoding matched the reference tokenizer on 111 validation inputs, including both thinking-enabled and thinking-disabled chat prompts, Unicode, whitespace, and random strings. The reference package was used only as a development oracle, not by the delivered test script.

Downloads are pinned to one commit per run and checked against the repository's LFS SHA-256 or Git blob hash. Completed files are reused after verification; interrupted weight downloads resume from `.part` files. The script supports the single checkpoint file in this release and sharded indexes if the repository supplies one. The default directory `checkpoints/LFM3-1.7B/` is ignored by Git and includes the model license. `--offline` verifies the recorded local files without contacting the server.

The real run on **2026-10-28** loaded **286 checkpoint tensors**, verified all **1,728,801,536 unique parameters** and tied embeddings, and generated **819 new tokens in approximately 53.7 seconds on CUDA with BF16 weights**, ending on EOS before the configured 2,048-token limit. It used revision **`40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`**; every generated step had finite logits. The generated answer discusses the nematode and its use as a model organism. This verifies end-to-end local inference; generated factual claims are not a model-quality benchmark.

The streaming/speculative validation run generated **819 tokens in 50.05 seconds**, or **16.36 tok/s**, on CUDA/BF16 and reached EOS. It accepted **47 of 295 proposed draft tokens** and used **772 target forward passes**, rather than one per emitted token. The live stream was observed before completion. These are measurements from that run, not a controlled speedup comparison against a separately timed baseline.

An additional pretrained comparison used the pasted official computational classes with the same checkpoint tensors. Last-position logits matched **exactly** with the same PyTorch SDPA implementation and mask conventions at prefix lengths **17, 49, and 273**, including a prefix longer than the sliding window. The official eager-attention path showed BF16 rounding differences (maximum absolute logit differences 0.1875–0.375) but selected the same highest-scoring next token at all three prefixes. These are historical checks from the original implementation; subsequent generation runs replace the result JSON.

The complete result is saved to `checkpoints/LFM3-1.7B/generation_result.json`: prompt and generated IDs, raw decoded continuation, separately parsed `reasoning_content` and `assistant_content`, sampling/speculation settings, throughput metrics, revision, stop reason, and verification flags. Standard output streams reasoning and the assistant answer; tests and the performance summary go to standard error. If a short token limit ends during thinking, the JSON preserves the partial reasoning and records an empty final answer. The previous result was retained as `generation_result_before_streaming.json` when the streaming implementation was validated.

`python test_lfm3.py --unit` runs 25 offline tests, including sampling order, EOS/length/context stopping, inference without gradients, download resumption/checksums, offline reuse, reasoning/content separation, and CLI routing. Streaming checks verify early flushes and split UTF-8 characters. Speculative checks cover block acceptance, rejection without leaking draft tokens, fewer target forwards on repeated spans, and agreement with sequential sampling across ten seeds using repetition penalties. `python -m unittest test_lfm3` also discovers the tests. Importing `test_lfm3.py` does not run tests, download files, or start generation.
