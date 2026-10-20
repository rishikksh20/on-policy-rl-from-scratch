# Real LFM2.5-230M • PyTorch-only On-Policy Reinforcement Learning 

Six `.py` trainers using shared model code in `lfm.py`, no TRL, veRL, Hugging Face Transformers, `safetensors` package, or third-party RL framework in their training code. They reimplement the **real hybrid LFM2.5-230M architecture in PyTorch**, not a toy GRU (8 double-gated short convolution layers + 6 grouped-query attention layers, 14 layers in total, RMSNorm, SwiGLU, RoPE, 65,536 vocab, 229,693,184 unique model parameters). The raw BF16 HF checkpoint is read with a small standard-library safetensors reader; weights are loaded to FP32 for a simple training example.

**What I verified:** All six `--tiny` training loops run and produce differentiable updates, independent actor/critic backbones load a synthetic checkpoint and update separately, GRPO/RLVR show groups with positive/negative advantages, preference-reward GRPO updates the actor while keeping the reward model frozen, model architecture size computes to exactly 229,693,184 parameters. **What I could not test here:** a forward/training run with the real 459 MB HF checkpoint (not downloaded in this environment) or decoding the actual tokenizer.json. See caveats below. Do not treat this as production-grade RL training.

## 1. Files

| Script | Training goal | Policy/critic/reward |
|---|---|---|
| `01_ppo_deterministic.py` | Objective arithmetic / exact-match tasks | Shared LFM actor/critic backbone, deterministic verifier |
| `02_ppo_preference_reward.py` | Subjective preferences | Shared LFM actor/critic backbone + explicitly trained LFM-based scalar reward head |
| `03_grpo.py` | Group-relative RL | LFM actor, **no critic**, deterministic scores |
| `04_rlvr_code_verifier.py` | RLVR coding | LFM actor + GRPO optimizer, AST-checked unit tests |
| `05_ppo_independent_critic.py` | Exact-match PPO with an independent critic | Separate pretrained LFM actor and critic backbones, deterministic verifier |
| `06_grpo_preference_reward.py` | Group-relative RL from subjective preferences | LFM actor + trained, frozen preference reward model; **no critic** |
| `lfm.py` | Shared model implementation | LFM architecture, policy/value heads, configuration, checkpoint loading, model initialization |
| `prepare_data_optional.py` | Turn human-readable prompts into model token IDs | This **one-time optional** step uses the official Transformers tokenizer; trainers don't depend on it |

All six trainers import their model implementation from `lfm.py`; keep it alongside the scripts. Dataset loading, sampling, rewards, and RL losses stay in the trainers.

## 2. Requirements and run commands

Install **only PyTorch** to train:

```bash
pip install torch

# Fast architecture + loss smoke tests, small RANDOM LFM (NOT pretrained!):
python 01_ppo_deterministic.py --tiny --steps 3 --lr 0.001
python 02_ppo_preference_reward.py --tiny --steps 3 --lr 0.001
python 03_grpo.py --tiny --steps 3 --group-size 4 --lr 0.001
python 04_rlvr_code_verifier.py --tiny --steps 3 --group-size 4 --lr 0.001
python 05_ppo_independent_critic.py --tiny --steps 3 --lr 0.001 --critic-lr 0.001
python 06_grpo_preference_reward.py --tiny --steps 3 --group-size 4 --lr 0.001
python tests_smoke.py
```

For *actual* LFM2.5-230M training, get the 459 MB official checkpoint and tokenizer locally:

```bash
mkdir -p checkpoint
curl -L https://huggingface.co/LiquidAI/LFM2.5-230M/resolve/main/model.safetensors -o checkpoint/model.safetensors
curl -L https://huggingface.co/LiquidAI/LFM2.5-230M/resolve/main/tokenizer.json -o checkpoint/tokenizer.json

# OPTIONAL and only required for converting textual datasets to token IDs:
pip install 'transformers>=5.2'
python prepare_data_optional.py

# PPO with deterministic / exact-match verifier
python 01_ppo_deterministic.py --checkpoint checkpoint/model.safetensors --data tokenized/numeric_tokens.jsonl --steps 20 --max-new 12 --lr 1e-5

# PPO with separate pretrained LFM actor and critic backbones
python 05_ppo_independent_critic.py --checkpoint checkpoint/model.safetensors --data tokenized/numeric_tokens.jsonl --steps 20 --max-new 12 --lr 1e-5 --critic-lr 1e-5

# PPO with explicit learned preference reward
python 02_ppo_preference_reward.py --checkpoint checkpoint/model.safetensors --data tokenized/preferences_tokens.jsonl --rm-epochs 10 --steps 20 --max-new 24 --lr 1e-5

# GRPO with exact-match verifier
python 03_grpo.py --checkpoint checkpoint/model.safetensors --data tokenized/numeric_tokens.jsonl --group-size 4 --steps 20 --max-new 12 --lr 1e-5

# GRPO with a learned preference reward model, using the same data as PPO variant 2
python 06_grpo_preference_reward.py --checkpoint checkpoint/model.safetensors --data tokenized/preferences_tokens.jsonl --rm-epochs 10 --group-size 4 --steps 20 --max-new 24 --lr 1e-5

# RLVR where the model writes arithmetic expressions; AST verifier runs unit tests
python 04_rlvr_code_verifier.py --checkpoint checkpoint/model.safetensors --tokenizer checkpoint/tokenizer.json --data tokenized/coding_tokens.jsonl --group-size 4 --steps 20 --max-new 24 --lr 1e-5
```

`--tiny` changes the architecture to a three-layer small random network and constrains its output to a **four-token action space** to make the gradients visible. It uses synthetic token IDs with no meaningful text semantics. **Do not combine `--tiny` with the 230M checkpoint.** Real mode trains the 65,536-vocabulary LFM architecture on the full sampled-token distribution. Sampling is deliberately slow because no KV cache is implemented.

**Checkpoint choice**: we use the instruction-tuned `LiquidAI/LFM2.5-230M` policy. The base checkpoint (`LiquidAI/LFM2.5-230M-Base`) is better suited to performing SFT *before* PPO/GRPO. Their architectures match, but direct RL on a base-only checkpoint is generally more difficult.

## 3. Dataset formats, precisely

Format is **JSON Lines** (`.jsonl`): one JSON object per line. The RL scripts take *token IDs* as their inputs to keep the training dependency exactly `torch`. The standalone **optional** preparation script applies the real model's chat template/tokenizer.

### 3A. Text-level SFT input, for comparison

```json
{"messages":[{"role":"user","content":"What is 2+3?"},{"role":"assistant","content":"5"}]}
```

For SFT, the targets are assistant tokens and the prompt positions are masked with `-100` in shifted cross-entropy. This package implements post-training RL only, not an SFT trainer.

### 3B. Deterministic PPO / GRPO text input

```json
{"prompt":"What is 2+3? Answer with only the number.","answer":"5"}
```

The optional preprocessor transforms it into **numeric token IDs** (values below are illustrative ONLY, **not** genuine LFM IDs):

```json
{"prompt_ids":[1,240,1025,501,12],"answer_ids":[369,7]}
```

`prompt_ids`: the entire conversation header including the assistant-generation prefix. `answer_ids`: the desired assistant continuation including final EOS (`7` in this LFM config). This target is used by the **reward verifier**, not by a cross-entropy loss. The actor samples its own response tokens; if its sampled completion matches the target token sequence (excluding one trailing EOS), it receives `+1`, otherwise `0`.

The model may generate a correct mathematical answer in a different tokenization or phrasing and still get zero: this **strict token-level verifier is deliberately simplistic**. In practical numeric RLVR, decode the text, parse the number, and compare numerical values, not token IDs.

### 3C. Preference dataset for learned reward (scripts 02 and 06)

Human-readable:

```json
{"prompt":"A delivery is late. Respond to a frustrated customer.","chosen":"I'm sorry about the delay. Let me help track it.","rejected":"Stop complaining."}
```

Model-ready token JSONL (the following IDs are deliberately fake examples to show structure):

```json
{"prompt_ids":[1,640,12],"chosen_ids":[72,551,16,7],"rejected_ids":[33,22,7]}
```

`chosen_ids` / `rejected_ids` are preferred and dispreferred *assistant continuations*, respectively. **These are NOT scalar reward targets.** The reward model learns a scalar from pairs via a pairwise preference loss.

**Reward model dataset vs PPO dataset**: Pairwise human-labeled rows train the reward model first. For PPO, only `prompt_ids` are used to generate new model responses. The trained reward model scores those *new* responses. The chosen/rejected examples are not copied into PPO rollouts.

### 3D. RLVR coding dataset

Human-readable:

```json
{"prompt":"Write only the Python expression using a and b that adds them.","tests":[{"a":2,"b":3,"expected":5},{"a":-4,"b":6,"expected":2}]}
```

Model-ready structure:

```json
{"prompt_ids":[1,810,14,77],"tests":[{"a":2,"b":3,"expected":5},{"a":-4,"b":6,"expected":2}]}
```

The model generates tokens; the decoder converts them to source code (e.g., `a+b`). The verifier parses a **small whitelist of arithmetic Python syntax** with `ast` and checks each test. The verifier returns a **binary scalar**, `1` for ALL tests passing, `0` otherwise. No target source-code string is required. It supports only a bare Python expression or `def solve(a,b): return ...` with `+`, `-`, `*`, constants and variables. **No arbitrary Python execution** is performed; a real coding RLVR setup should use a separate sandbox/container with resource limits and trustworthy tests.

## 4. PPO objective: actor, critic, and their exact targets

At token index `t`, state `s_t` is prompt + previously sampled output tokens. The sampled next token `a_t` is the action.

Rollout buffer structure:

```json
{
  "prompt_ids": [1, 100, 200],
  "actions": [11, 12],
  "old_log_probs": [-1.20, -0.60],
  "old_values": [0.20, 0.60],
  "rewards": [0.00, 1.00]
}
```

The log probabilities and value predictions come from model inference on **those exact states and actions**. `rewards[-1]` includes the final task reward, plus the per-token reference-policy KL penalty.

**1. Sample action tokens** with `sample_tokens(policy, prompt)`, then compute `old_logp` and `old_values` via `positions_log_probs`. Under `torch.no_grad()`, these tensors cannot change during PPO updates.

**2. Construct per-token rewards** with:

\[
r_t = -\beta\,\left[\log\pi_{\mathrm{old}}(a_t\mid s_t)-\log\pi_{\mathrm{ref}}(a_t\mid s_t)\right]
\]

Then `r_last += task_score`. The log-prob ratio is a **sampled** KL penalty; it can be negative on individual trajectories, although its expectation has a KL interpretation.

**3. Compute GAE:**

\[
\delta_t = r_t + \gamma V_{\mathrm{old}}(s_{t+1}) - V_{\mathrm{old}}(s_t)
\]
\[
A_t = \delta_t + \gamma\lambda A_{t+1},\qquad R_t=A_t+V_{\mathrm{old}}(s_t)
\]

`R_t` is the **training target for the critic**. It is computed from observed rewards and earlier critic predictions; it is not a human-authored label. `gae()` receives zero bootstrap at both EOS and the token limit: each ends the scored-answer task after the outcome reward is assigned.

**4. Calculate PPO losses:**

\[
\rho_t(\theta)=\exp[\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)]
\]
\[
L_{\mathrm{actor}}=-\mathbb{E}_t\left[\min\left(\rho_t A_t,\; \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right]
\]
\[
L_{\mathrm{critic}}=\tfrac12\mathbb{E}_t[(V_\phi(s_t)-R_t)^2]
\]
\[
L_{\mathrm{PPO}}=L_{\mathrm{actor}}+c_v L_{\mathrm{critic}}
\]

In scripts **01 and 02**, actor and critic share the LFM backbone and optimize it through two heads. An explicit value head `Linear(hidden_size, 1)` is **newly initialized**, and not present in the LFM checkpoint.

Script **05** uses the same deterministic reward, dataset format, GAE equations, and clipped actor loss as script 01, with an **independent critic**:

```text
prompt + previous response tokens
    ├── LFMPolicy: own pretrained LFM backbone + tied LM head → next-token logits
    └── LFMCritic: own pretrained LFM backbone + scalar head → expected future return
```

Both backbones load `--checkpoint` into separate trainable parameters. The critic has no vocabulary head, and its scalar value head starts newly initialized. `positions_values` aligns each value with the state **before** the corresponding sampled action. EOS and the token limit both end this scored-answer task and use zero bootstrap.

Two AdamW optimizers update the actor and critic independently. The actor minimizes `L_actor`; the critic minimizes `c_v * L_critic` (`c_v=0.5`, matching script 01). `--lr` controls the actor and `--critic-lr` controls the critic (default: `--lr`). Rollout advantages and critic targets remain fixed throughout the PPO epochs. The frozen reference model is a separate actor copy used for KL rewards. This uses an additional trainable backbone compared with shared-head PPO.

## 5. PPO with explicit **preference reward model**

**Step 1 — Train reward model from chosen/rejected pairs.** Use the **frozen pretrained LFM backbone** to extract an end-of-response embedding; train an added scalar linear head (`LearnedReward.head`) with:

\[
P(y^+\succ y^-\mid x)=\sigma(r_\psi(x,y^+)-r_\psi(x,y^-))
\]
\[
L_{\mathrm{RM}}=-\log\sigma(r_\psi(x,y^+)-r_\psi(x,y^-))
=\operatorname{softplus}(r_\psi(x,y^-)-r_\psi(x,y^+))
\]

For example, if `score_chosen=+2.0`, `score_rejected=-1.0`, the loss is `softplus(-3) ≈ 0.0486`. The RM improves when preferred answers receive higher scores. **This is a real learned reward head**; it is deliberately shallow and only trains the last layer because the goal is to show pairwise supervision with a small memory footprint. Full-reward-model training would additionally unfreeze its backbone.

**Step 2 — PPO samples its own completions**. The reward model scores them with `score=tanh(reward_model(prompt,sampled_response)-score_center)` to keep the example bounded. `score_center` is the fixed mean raw score of labeled chosen/rejected responses after reward-head training, which removes a shared additive score offset. This is centering on training examples, not held-out calibration; reward scale and saturation still need monitoring. This scalar is inserted at the final response token and fed to `gae`, then `ppo_update` runs the exact same actor/critic losses as before. The RM is frozen throughout PPO optimization, so only actor/critic change.

**Do not** train the reward model on its own PPO-generated rewards without independent preference labels; that would create a self-reinforcing score loop.

## 6. GRPO objective — no critic

Script **03** uses deterministic exact-match rewards. Script **06** uses the same `LearnedReward` architecture and pairwise training as script 02, then freezes that model and scores fresh sampled completions with its **raw scalar output**. Chosen/rejected labels train only the reward head; GRPO trains the actor on its own sampled responses to those prompts. There is no value head, GAE, or critic optimizer. `--rm-epochs` and `--rm-lr` control reward-head training; `--lr` controls the actor. The frozen reference policy still supplies the KL penalty.

Given the same prompt, sample a group `G` of independent responses, each with outcome reward `r_i`. Then:

\[
\mu=\frac1G\sum_i r_i,\qquad\sigma=\sqrt{\frac1G\sum_i(r_i-\mu)^2}
\]
\[
A_i=\frac{r_i-\mu}{\sigma+10^{-6}}
\]

For `rewards=[1,0,1,0]`, mean `0.5`, std `0.5`, hence `advantages≈[+1,-1,+1,-1]`. These group-relative advantages are applied to *every sampled token in its response*. No critic/value regression is used.

GRPO then minimizes a PPO-style **clipped policy loss** and a reference KL penalty:

\[
L_{\mathrm{GRPO}}=-\frac1G\sum_i\frac1{|y_i|}\sum_t \min(\rho_{it}A_i,\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i) +\beta D_{\mathrm{KL}}
\]

The scripts use a nonnegative sampled KL estimator:

\[
\hat D=\exp(\log\pi_{\mathrm{ref}}-\log\pi_{\theta}) - (\log\pi_{\mathrm{ref}}-\log\pi_{\theta}) -1
\]

If every rollout in a group gets the *same* reward, all `A_i=0`; then there is no outcome-policy learning signal (only any reference KL penalty remains). This is a fundamental sparse-reward issue, not a code bug.

## 7. RLVR objective — verifiable rewards + GRPO

**RLVR does not define a new optimizer.** It means rewards are verified from objective evidence (unit tests, exact answers, a math evaluator, a tool outcome). The RLVR script calls `verify_unit_tests(generated_code,tests)`:

\[
r(q,y)=\begin{cases}1,&\text{code is syntactically allowed and passes every test}\\0,&\text{otherwise.}\end{cases}
\]

Then sample G codes, derive GRPO group-relative advantages, and use the **same GRPO clipped-policy update as script 03**. Code verifier handles **task success**; GRPO handles **gradient optimization**. There is no supervised target string in the RLVR dataset.

## 8. Walk through the functions

| Function | Role |
|---|---|
| `LFMPolicy.forward` | Embedding → 14 LFM layers → RMSNorm → tied LM logits; optional scalar values |
| `LFMCritic.forward`, `initialize_critic` | Independent pretrained LFM backbone → scalar value per position; no LM head |
| `read_safetensors`, `load_weights` | Pure-Python HF BF16 weight reading and strict shape/name checking |
| `sample_tokens` | Sample actions autoregressively using `torch.multinomial` |
| `positions_values` (script 05) | Align independent critic values with the actor's pre-action token prefixes |
| `positions_log_probs` | Shift next-token logits so actions get log π(a_t|s_t); optional pre-action critic values |
| `gae` | Reward + predicted-next-value → TD residual → advantages and critic returns |
| `rollout` | Cache old policy, frozen reference probs, rewards and critic targets |
| `ppo_update` | Clipped actor + value regression, backpropagation and clipping |
| `LearnedReward.forward` | Score prompt + generated/annotated answer using pretrained backbone and trainable scalar head |
| `train_reward_model` | Update reward head on chosen/rejected preference data |
| `group_rollout` | Sample G candidates for one prompt, get rewards and normalize into GRPO advantages |
| `grpo_update` | Clipped policy and reference KL without critic |
| `Decoder.decode` | Convert actual tokenizer IDs into generated program text for coding verification |
| `extract_expression`, `arithmetic`, `verify_unit_tests` | Parse/interpret a tiny safe expression grammar and compare unit-test outputs |
| `verified_group` | Group sampling + unit-test scoring + normalized advantages |
| `update_with_grpo` | GRPO updates using verified coding rewards |

## 9. Real-world caveats and suggested improvements

1. **Real checkpoint integration is structurally matched but not integration-tested here.** Confirm actual `model.safetensors` strict loading and decoded outputs against the official Transformers inference baseline before investing in training.
2. **Tokenizer fidelity matters.** The optional data-preparation file uses the official chat template; RLVR has only a lightweight **decoder** for common ByteLevel/Metaspace tokenizers, not a complete reproduction of `tokenizer.json`'s postprocessing. Validate round-trip decoding with official tokenizer. Any mismatch corrupts coding rewards.
3. **A small toy dataset will not make pretrained LFM learn general behavior.** Use hundreds/thousands of training problems, held-out evaluation, sensible lengths, and multiple rollouts.
4. **True sparse rewards are hard**: random completions seldom pass exact-match checks. Add curriculum, stronger SFT initialization, reasonable sample lengths and tasks with solvable outputs, and report success/variance rates.
5. These examples use **batch=1, no KV cache, serial sampling, FP32 AdamW, no activation checkpointing or adaptive KL controller**. They are intentionally inefficient. Full PPO + reference + RM + critic can require significant VRAM; use appropriately sized devices and reduce lengths if needed.
6. **Token limits:** PPO and GRPO score the response as finished at EOS or the token limit. PPO therefore uses zero bootstrap for both. A genuinely interrupted continuing task would need different reward timing and a nonzero bootstrap. For real RLVR, decide explicitly if unfinished code should receive 0.
7. PPO/GRPO objectives here are **pedagogical implementations**, not drop-in reproductions of any particular RL paper's complete hyperparameter choices, credit assignment, loss masking, and distributed sampling.

References: [LFM2.5-230M checkpoint](https://huggingface.co/LiquidAI/LFM2.5-230M), [base](https://huggingface.co/LiquidAI/LFM2.5-230M-Base), [Hugging Face LFM2 architecture implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py).
