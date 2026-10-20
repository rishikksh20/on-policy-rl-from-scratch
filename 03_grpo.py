"""LFM2.5-230M educational PyTorch implementation (architecture-compatible, no caching).

Requires torch + Python standard library, no Transformers/TRL/veRL/safetensors package.
The pretrained model is stored in a local model.safetensors (download separately).
You must supply token IDs produced by the OFFICIAL LFM tokenizer; numbers in demo
JSONL are illustrations, not text-to-token conversions.

Model architecture and checkpoint loading live in lfm.py.
Architecture follows HF Lfm2ForCausalLM: 8 gated depthwise conv, 6 GQA,
SwiGLU, RMSNorm, rotary positions, weight-tied LM head. No KV-cache,
packing, distributed training, or inference optimization.
"""
import argparse
import json

import torch
from torch import nn
import torch.nn.functional as F

from lfm import initialize


def load_jsonl(path):
    """JSON Lines: one dict per line, numeric token ids from OFFICIAL tokenizer."""
    with open(path,encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def positions_log_probs(policy,prompt,actions,critic=False,allowed_ids=None):
    """Teacher-force *same sampled actions* to recover per-token log π(a_t|s_t).

    Full input: prompt + actions[:-1] -> logits at indices P-1 .. P+T-2.
    Crucial causal shift: targets are actions[0..T-1].
    Returns log_probs [T], values [T] (pre-action states), or None.
    """
    device=next(policy.parameters()).device
    ids=torch.tensor([prompt+actions[:-1]],dtype=torch.long,device=device)
    logits, values=policy(ids,with_values=critic)
    offset=len(prompt)-1
    selection=logits[:,offset:offset+len(actions),:].float()
    if allowed_ids is not None:
        # Restricted action-space demonstration: normalize ONLY among these ids.
        selection=selection[:,:,allowed_ids]
        mapped=[allowed_ids.index(a) for a in actions]
    else:
        mapped=actions
    act=torch.tensor(mapped,dtype=torch.long,device=device).view(1,-1,1)
    logp=F.log_softmax(selection,dim=-1).gather(-1,act).squeeze(-1).squeeze(0)
    val=values[:,offset:offset+len(actions)].squeeze(0) if critic else None
    return logp,val


@torch.no_grad()
def sample_tokens(policy,prompt,max_new=6,eos=7,temperature=1.0,allowed_ids=None):
    """Autoregressive multinomial rollout; deliberately no KV cache for clarity."""
    device=next(policy.parameters()).device
    actions=[]
    for _ in range(max_new):
        ids=torch.tensor([prompt+actions],dtype=torch.long,device=device)
        logits,_=policy(ids)
        scores=logits[0,-1].float()
        if allowed_ids is not None:
            scores=scores[allowed_ids]
        p=torch.softmax(scores/temperature,dim=-1)
        ix=torch.multinomial(p,1).item()
        a=allowed_ids[ix] if allowed_ids is not None else ix
        actions.append(a)
        if a==eos:
            break
    return actions


def token_reward(generated,answer,eos=7):
    """Objective verifier: exact token sequence (after removing trailing EOS)."""
    def clean(xs):
        xs=list(xs)
        return xs[:-1] if xs and xs[-1]==eos else xs
    return float(clean(generated)==clean(answer))


def show_rollout(step,prompt,completion,score,loss):
    print(f'step={step} prompt_ids={prompt[:5]} response_ids={completion[:8]} '
          f'reward={score:+.2f} loss={loss:.4f}')


def common_args(description):
    a=argparse.ArgumentParser(description=description)
    a.add_argument('--checkpoint',help='Path to official model.safetensors')
    a.add_argument('--data',help='Pre-tokenized JSONL data; default: tiny synthetic demo')
    a.add_argument('--tiny',action='store_true',help='Small RANDOM model for debugging, NOT 230M')
    a.add_argument('--steps',type=int,default=2)
    a.add_argument('--max-new',type=int,default=16)
    a.add_argument('--lr',type=float,default=1e-5)
    a.add_argument('--seed',type=int,default=42)
    return a



# ================================================================
# 03. GRPO: GROUP RELATIVE Policy Optimization; NO CRITIC.
# One prompt -> G independently sampled completions -> G rewards.
# A_i=(r_i-mean(r_group))/(std(r_group)+epsilon).
# Same A_i is applied to all generated tokens of response i.
# Clipped importance sampling prevents excessive policy shifts.
# ================================================================
import copy


@torch.no_grad()
def group_rollout(actor,reference,record,group_size,max_new,allowed):
    """Collect G response trajectories from ONE prompt and freeze logprobs."""
    prompt=record['prompt_ids']
    out=[]
    for _ in range(group_size):
        actions=sample_tokens(actor,prompt,max_new=max_new,allowed_ids=allowed)
        old_lp,_=positions_log_probs(actor,prompt,actions,allowed_ids=allowed)
        ref_lp,_=positions_log_probs(reference,prompt,actions,allowed_ids=allowed)
        r=token_reward(actions,record['answer_ids'])
        out.append(dict(actions=actions,old_lp=old_lp.detach(),
                        ref_lp=ref_lp.detach(),reward=r))
    r=torch.tensor([o['reward'] for o in out],device=out[0]['old_lp'].device)
    # Population standard deviation; all-equal groups must have A=0.
    std=r.std(unbiased=False)
    advantage=(r-r.mean())/(std+1e-6)
    for i,o in enumerate(out):
        o['advantage']=advantage[i].detach()
    return out


def grpo_update(actor,optimizer,prompt,group,allowed,
                clip_eps=0.2,beta=0.02,epochs=2):
    """GRPO objective for individual output tokens.

    ratio_(i,t) = exp(logpi_new - logpi_old)
    L_clip_i = -mean_t[min(ratio*A_i, clip(ratio,1-e,1+e)*A_i)]
    D_KL_i = mean_t[exp(logpi_ref-logpi_new)
                      -(logpi_ref-logpi_new)-1] (nonnegative estimator)
    L = mean_i[L_clip_i + beta*D_KL_i]

    This implementation uses response-level normalized rewards. If the
    entire group receives the same reward, policy advantages are zero.
    """
    for _ in range(epochs):
        losses=[]
        for item in group:
            new_lp,_=positions_log_probs(actor,prompt,item['actions'],allowed_ids=allowed)
            ratio=(new_lp-item['old_lp']).exp()
            A=item['advantage']
            p=-torch.minimum(ratio*A,ratio.clamp(1-clip_eps,1+clip_eps)*A).mean()
            log_ref_ratio=item['ref_lp']-new_lp
            kl=(log_ref_ratio.exp()-log_ref_ratio-1).mean()
            losses.append(p+beta*kl)
        loss=torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step()
    return loss.item()


def main():
    a=common_args('LFM2.5-230M GRPO using torch only')
    a.add_argument('--group-size',type=int,default=4)
    args=a.parse_args()
    torch.manual_seed(args.seed)
    actor,device=initialize(args.tiny,args.checkpoint,critic=False)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    opt=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    if args.data:
        data=load_jsonl(args.data)
    elif args.tiny:
        data=[{'prompt_ids':[1,22,33], 'answer_ids':[12]}]
    else:
        raise ValueError('Supply --data numeric.jsonl for real LFM training')
    allowed=[10,11,12,13] if args.tiny else None
    max_new=1 if args.tiny else args.max_new
    for step in range(args.steps):
        sample=data[step%len(data)]
        group=group_rollout(actor,reference,sample,args.group_size,max_new,allowed)
        loss=grpo_update(actor,opt,sample['prompt_ids'],group,allowed)
        print(f'GRPO step={step} loss={loss:.4f} rewards={[r["reward"] for r in group]} '
              f'advantages={[round(g["advantage"].item(),3) for g in group]}')
        print('    sampled_token_ids=',[g['actions'] for g in group])


if __name__=='__main__':
    main()
