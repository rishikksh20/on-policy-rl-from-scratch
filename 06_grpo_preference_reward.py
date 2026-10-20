"""LFM2.5-230M GRPO with a learned pairwise preference reward model.

Requires torch + Python standard library. Model code lives in lfm.py.
Train a reward head on prompt/chosen/rejected token IDs, then freeze it.
Sample multiple responses per prompt and normalize the reward-model scores.
The actor learns through GRPO; there is no critic or deterministic verifier.

    python 06_grpo_preference_reward.py --tiny --steps 3 --group-size 4 --lr 0.001
"""
import argparse
import copy
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


# ================================================================
# 06. GRPO with an EXPLICIT LEARNED PREFERENCE REWARD MODEL.
# Pairwise preference labels train the RM, not the actor directly.
# Frozen RM scores G fresh completions of the SAME prompt.
# Group-relative advantages replace PPO's learned value critic.
# ================================================================


@torch.no_grad()
def group_rollout(actor,reference,rm,prompt,group_size,max_new,allowed):
    """Score sampled completions with a frozen RM and cache GRPO advantages."""
    out=[]
    for _ in range(group_size):
        actions=sample_tokens(actor,prompt,max_new=max_new,allowed_ids=allowed)
        old_lp,_=positions_log_probs(actor,prompt,actions,allowed_ids=allowed)
        ref_lp,_=positions_log_probs(reference,prompt,actions,allowed_ids=allowed)
        # Raw preference score for the FULL sampled response; no verifier labels.
        reward=rm(prompt,actions).item()
        out.append(dict(actions=actions,old_lp=old_lp.detach(),
                        ref_lp=ref_lp.detach(),reward=reward))
    rewards=torch.tensor([item['reward'] for item in out],device=out[0]['old_lp'].device)
    advantages=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
    for item,advantage in zip(out,advantages):
        item['advantage']=advantage.detach()
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
    a=common_args('LFM2.5-230M GRPO with learned pairwise preference rewards')
    a.add_argument('--group-size',type=int,default=4)
    a.add_argument('--rm-epochs',type=int,default=3)
    a.add_argument('--rm-lr',type=float,default=1e-3,help='Preference reward-head learning rate')
    args=a.parse_args()
    if args.group_size<2:
        raise ValueError('--group-size must be at least 2 for group-relative advantages')
    if args.rm_epochs<1:
        raise ValueError('--rm-epochs must be at least 1 to train the preference reward model')
    if args.max_new<1:
        raise ValueError('--max-new must be at least 1')
    torch.manual_seed(args.seed)
    actor,device=initialize(args.tiny,args.checkpoint,critic=False)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    if args.data:
        prefs=load_jsonl(args.data)
    elif args.tiny:
        prefs=[{'prompt_ids':[1,22,33],'chosen_ids':[10],'rejected_ids':[11]},
               {'prompt_ids':[1,34,35],'chosen_ids':[10],'rejected_ids':[12]}]
    else:
        raise ValueError('Supply --data preference.jsonl with prompt/chosen/rejected token ids')
    if not prefs:
        raise ValueError('Preference data must contain at least one record')
    rm=LearnedReward(actor,cfg(args.tiny))
    train_reward_model(rm,prefs,epochs=args.rm_epochs,lr=args.rm_lr)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    allowed=[10,11,12,13] if args.tiny else None
    max_new=1 if args.tiny else args.max_new
    for step in range(args.steps):
        prompt=prefs[step%len(prefs)]['prompt_ids']
        group=group_rollout(actor,reference,rm,prompt,args.group_size,max_new,allowed)
        loss=grpo_update(actor,optimizer,prompt,group,allowed)
        print(f'GRPO preference step={step} loss={loss:.4f} '
              f'rewards={[round(item["reward"],3) for item in group]} '
              f'advantages={[round(item["advantage"].item(),3) for item in group]}')
        print('    sampled_token_ids=',[item['actions'] for item in group])


if __name__=='__main__':
    main()
