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

from lfm import cfg, initialize


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
# 02. PPO with an EXPLICIT LEARNED REWARD MODEL for subjective tasks.
# Pairwise dataset: chosen and rejected responses, NOT scalar labels.
# RM: s(prompt,response) -> scalar; Bradley-Terry preference loss.
# Actor: PPO sampled responses, critic: GAE return prediction.
# ================================================================
import copy


class LearnedReward(nn.Module):
    """Frozen LFM feature extractor + trainable reward head.

    This is a real *trained* reward model, not an LLM-as-judge API.
    Freeze backbone to keep the tutorial affordable. Production systems
    may fine-tune the whole reward backbone; see limitations in README.
    """
    def __init__(self,pretrained_actor,c):
        super().__init__()
        self.backbone=copy.deepcopy(pretrained_actor.model).eval().requires_grad_(False)
        self.head=nn.Linear(c['dim'],1).to(next(pretrained_actor.parameters()).device)

    def forward(self,prompt,response):
        # Head sees the FULL prompt and candidate answer representation.
        device=next(self.head.parameters()).device
        ids=torch.tensor([prompt+response],device=device)
        with torch.no_grad():
            last=self.backbone(ids)[:,-1].detach()
        return self.head(last).squeeze()


def train_reward_model(rm,records,epochs=5,lr=1e-3):
    """Pairwise preference training, no numeric labels necessary.

      p(chosen > rejected) = sigmoid(score_chosen-score_rejected)
      loss_RM = -log(sigmoid(score_chosen-score_rejected))
              = softplus(score_rejected-score_chosen).
    """
    optimizer=torch.optim.AdamW(rm.head.parameters(),lr=lr)
    for epoch in range(epochs):
        for record in records:
            p=record['prompt_ids']
            chosen=rm(p,record['chosen_ids'])
            rejected=rm(p,record['rejected_ids'])
            loss=F.softplus(rejected-chosen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f'reward_model epoch={epoch+1} preference_loss={loss.item():.4f} '
              f'score_chosen={chosen.item():+.3f} score_rejected={rejected.item():+.3f}')
    rm.eval().requires_grad_(False)


def gae(rewards,values,bootstrap,gamma=0.99,lam=0.95):
    """Return[t] = Advantage[t] + OLD value[t]."""
    advantages=torch.zeros_like(rewards)
    acc=rewards.new_zeros(())
    next_v=bootstrap
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_v-values[t]
        acc=delta+gamma*lam*acc
        advantages[t]=acc
        next_v=values[t]
    return advantages,advantages+values


@torch.no_grad()
def rollout(actor,reference,rm,prompt,max_new,allowed,beta=0.02):
    """Use learned reward from FULL sampled completion; distribute at end."""
    actions=sample_tokens(actor,prompt,max_new=max_new,allowed_ids=allowed)
    old_lp,old_v=positions_log_probs(actor,prompt,actions,critic=True,allowed_ids=allowed)
    ref_lp,_=positions_log_probs(reference,prompt,actions,allowed_ids=allowed)
    score=torch.tanh(rm(prompt,actions)).item() # bound [-1,1] for toy stability
    rewards=-beta*(old_lp-ref_lp)
    rewards[-1]+=score
    if actions[-1]==7:
        boot=old_v.new_zeros(())
    else:
        ids=torch.tensor([prompt+actions],device=old_v.device)
        _,values=actor(ids,with_values=True)
        boot=values[0,-1].detach()
    advantages,returns=gae(rewards,old_v,boot)
    return dict(prompt=prompt,actions=actions,old_lp=old_lp.detach(),
                advantages=advantages.detach(),returns=returns.detach(),
                score=score)


def ppo_update(actor,opt,buffer,allowed,epochs=2,epsilon=0.2):
    """Actor-critic update identical to deterministic-reward PPO.

    Only the scalar reward *source* changed; the PPO optimizer did not.
    """
    for _ in range(epochs):
        new_lp,v=positions_log_probs(actor,buffer['prompt'],buffer['actions'],
                                     critic=True,allowed_ids=allowed)
        ratio=(new_lp-buffer['old_lp']).exp()
        A=buffer['advantages']
        actor_loss=-torch.minimum(ratio*A,ratio.clamp(1-epsilon,1+epsilon)*A).mean()
        critic_loss=.5*(v-buffer['returns']).square().mean()
        loss=actor_loss+0.5*critic_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.)
        opt.step()
    return loss.item(),actor_loss.item(),critic_loss.item()


def main():
    a=common_args('LFM2.5-230M PPO with learned pairwise preference reward model')
    a.add_argument('--rm-epochs',type=int,default=3)
    args=a.parse_args()
    torch.manual_seed(args.seed)
    actor,device=initialize(args.tiny,args.checkpoint,critic=True)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    if args.data:
        prefs=load_jsonl(args.data)
    elif args.tiny:
        prefs=[{'prompt_ids':[1,22,33],'chosen_ids':[10],'rejected_ids':[11]},
               {'prompt_ids':[1,34,35],'chosen_ids':[10],'rejected_ids':[12]}]
    else:
        raise ValueError('Supply --data preference.jsonl with prompt/chosen/rejected token ids')
    rm=LearnedReward(actor,cfg(args.tiny))
    train_reward_model(rm,prefs,epochs=args.rm_epochs)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    allowed=[10,11,12,13] if args.tiny else None
    max_new=1 if args.tiny else args.max_new
    for step in range(args.steps):
        p=prefs[step%len(prefs)]['prompt_ids']
        buf=rollout(actor,reference,rm,p,max_new,allowed)
        loss,apl,vpl=ppo_update(actor,optimizer,buf,allowed)
        show_rollout(step,p,buf['actions'],buf['score'],loss)
        print(f'    policy_loss={apl:.4f} critic_loss={vpl:.4f} '
              f'critic_targets={buf["returns"].tolist()}')


if __name__=='__main__':
    main()
