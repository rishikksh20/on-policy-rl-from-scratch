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

from lfm import LFMPolicy, cfg, initialize, read_safetensors


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
# 01. PPO with an OBJECTIVE / DETERMINISTIC reward.
# One rollout consists of a prompt and tokens sampled from LFM2.5.
# Reward = 1 for exact answer IDs, otherwise 0; terminal score at last token.
# Critic trained from GAE RETURNS, not from human-written value labels.
# ================================================================
import copy


def gae(rewards, old_values, bootstrap, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation (Schulman et al.).

    delta[t] = r[t] + gamma * V(s[t+1]) - V(s[t])
    A[t] = delta[t] + gamma * lam * A[t+1]
    critic_target[t] = A[t] + V_old(s[t]); all tensors frozen.

    bootstrap=0 for terminal EOS; use V(state_after_last_token)
    for truncated/nonterminal rollouts.
    """
    ans=torch.zeros_like(rewards)
    next_a=rewards.new_zeros(())
    next_v=bootstrap
    for t in range(len(rewards)-1,-1,-1):
        delta=rewards[t]+gamma*next_v-old_values[t]
        next_a=delta+gamma*lam*next_a
        ans[t]=next_a
        next_v=old_values[t]
    return ans,ans+old_values


@torch.no_grad()
def rollout(actor,reference,record,max_new,allowed,beta=0.02):
    """Cache old-policy and reference probabilities BEFORE gradient updates."""
    prompt=record['prompt_ids']
    actions=sample_tokens(actor,prompt,max_new=max_new,allowed_ids=allowed)
    old_lp,old_v=positions_log_probs(actor,prompt,actions,critic=True,allowed_ids=allowed)
    ref_lp,_=positions_log_probs(reference,prompt,actions,allowed_ids=allowed)
    score=token_reward(actions,record['answer_ids'])
    rewards=-beta*(old_lp-ref_lp)  # sampled KL regularization
    rewards[-1]+=score          # sparse final correctness reward
    if actions[-1] == 7:       # EOS means terminal
        bootstrap=old_v.new_zeros(())
    else:
        ids=torch.tensor([prompt+actions],device=old_v.device)
        _,all_v=actor(ids,with_values=True)
        bootstrap=all_v[0,-1].detach() # time limit is truncation, not terminal
    adv,returns=gae(rewards,old_v,bootstrap)
    return dict(prompt=prompt,actions=actions,old_lp=old_lp.detach(),
                advantages=adv.detach(),returns=returns.detach(),
                score=score,rewards=rewards.detach())


def ppo_update(actor,opt,batch,allowed,clip_eps=0.2,value_coef=0.5,epochs=2):
    """PPO actor-critic loss:

    ratio_t = exp(log_pi_new - log_pi_old)
    policy_loss = -mean(min(ratio*A,clip(ratio,1-e,1+e)*A))
    value_loss = .5 * mean((V_new - fixed_GAE_return)^2)
    total_loss = policy_loss + value_coef * value_loss
    """
    for _ in range(epochs):
        lp,values=positions_log_probs(actor,batch['prompt'],batch['actions'],
                                      critic=True,allowed_ids=allowed)
        ratio=(lp-batch['old_lp']).exp()
        advantage=batch['advantages']
        policy_loss=-torch.minimum(ratio*advantage,
            ratio.clamp(1-clip_eps,1+clip_eps)*advantage).mean()
        value_loss=.5*(values-batch['returns']).square().mean()
        loss=policy_loss+value_coef*value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.0)
        opt.step()
    return loss.item(),policy_loss.item(),value_loss.item()


def main():
    args=common_args('Pure PyTorch LFM2.5-230M PPO numeric reward').parse_args()
    torch.manual_seed(args.seed)
    actor,device=initialize(args.tiny,args.checkpoint,critic=True)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    # Only the actor + critic are optimized; reference is frozen.
    opt=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    if args.data:
        records=load_jsonl(args.data)
    elif args.tiny:
        records=[{'prompt_ids':[1,22,33], 'answer_ids':[10]}]
    else:
        raise ValueError('Supply --data numeric.jsonl for real LFM training')
    allowed=[10,11,12,13] if args.tiny else None
    max_new=1 if args.tiny else args.max_new
    print('PPO numeric | model:', 'tiny random smoke test' if args.tiny else 'REAL LFM2.5-230M')
    for step in range(args.steps):
        batch=rollout(actor,reference,records[step%len(records)],max_new,allowed)
        total,policy,value=ppo_update(actor,opt,batch,allowed)
        show_rollout(step,batch['prompt'],batch['actions'],batch['score'],total)
        print('   reward_vector=',batch['rewards'].tolist(),
              'critic_targets=',batch['returns'].tolist(),
              'advantages=',batch['advantages'].tolist(),
              f'policy_loss={policy:.3f} value_loss={value:.3f}')


if __name__=='__main__':
    main()
