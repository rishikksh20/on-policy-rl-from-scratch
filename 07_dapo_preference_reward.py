"""LFM2.5-230M DAPO with a learned pairwise preference reward model.

Requires only torch + Python standard library; shared model code is in lfm.py.
Train and freeze a reward head just as in 06_grpo_preference_reward.py.
DAPO adds decoupled clipping, dynamic sampling, token-level loss reduction,
and soft overlong reward shaping. There is no critic or reference-policy KL.
This is a preference-reward adaptation of https://arxiv.org/abs/2503.14476.

    python 07_dapo_preference_reward.py --tiny --steps 3 --group-size 4 --lr 0.001
"""
import argparse
import copy
import json
import math
import torch
from torch import nn
import torch.nn.functional as F
from lfm import cfg, initialize


def load_jsonl(path,vocab_size=65536):
    """Read official-tokenizer prompt/chosen/rejected IDs, as in script 06."""
    with open(path,encoding='utf8') as f:
        records=[json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError('Data must contain at least one record')
    for record in records:
        for key in ('prompt_ids','chosen_ids','rejected_ids'):
            ids=record.get(key) if isinstance(record,dict) else None
            if (not isinstance(ids,list) or not ids
                or any(type(i) is not int or not 0<=i<vocab_size for i in ids)):
                raise ValueError(f'{key} must contain nonempty integer token IDs in [0, {vocab_size})')
    return records


def positions_log_probs(policy,prompt,actions,critic=False,allowed_ids=None):
    """Recover log pi(a_t | prompt, actions[:t]) for the sampled actions."""
    device=next(policy.parameters()).device
    ids=torch.tensor([prompt+actions[:-1]],dtype=torch.long,device=device)
    logits,values=policy(ids,with_values=critic)
    offset=len(prompt)-1
    selection=logits[:,offset:offset+len(actions),:].float()
    if allowed_ids is not None:
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
    """Sample from exactly the distribution used by positions_log_probs."""
    if temperature!=1.0:
        raise ValueError('temperature must be 1.0 to match positions_log_probs')
    if max_new<1 or not prompt:
        raise ValueError('Sampling requires a nonempty prompt and max_new >= 1')
    device=next(policy.parameters()).device
    actions=[]
    for _ in range(max_new):
        ids=torch.tensor([prompt+actions],dtype=torch.long,device=device)
        logits,_=policy(ids)
        scores=logits[0,-1].float()
        if allowed_ids is not None:
            scores=scores[allowed_ids]
        ix=torch.multinomial(torch.softmax(scores,dim=-1),1).item()
        action=allowed_ids[ix] if allowed_ids is not None else ix
        actions.append(action)
        if action==eos:
            break
    return actions


class LearnedReward(nn.Module):
    """Frozen LFM backbone + scalar head trained on chosen/rejected pairs."""
    def __init__(self,pretrained_actor,c):
        super().__init__()
        self.backbone=copy.deepcopy(pretrained_actor.model).eval().requires_grad_(False)
        self.head=nn.Linear(c['dim'],1).to(next(pretrained_actor.parameters()).device)

    def forward(self,prompt,response):
        device=next(self.head.parameters()).device
        ids=torch.tensor([prompt+response],device=device)
        with torch.no_grad():
            last=self.backbone(ids)[:,-1].detach()
        return self.head(last).squeeze()


def train_reward_model(rm,records,epochs=5,lr=1e-3):
    """Fit softplus(score_rejected-score_chosen), then freeze the entire RM."""
    optimizer=torch.optim.AdamW(rm.head.parameters(),lr=lr)
    for epoch in range(epochs):
        for record in records:
            prompt=record['prompt_ids']
            chosen=rm(prompt,record['chosen_ids'])
            rejected=rm(prompt,record['rejected_ids'])
            loss=F.softplus(rejected-chosen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f'reward_model epoch={epoch+1} preference_loss={loss.item():.4f} '
              f'score_chosen={chosen.item():+.3f} score_rejected={rejected.item():+.3f}')
    rm.eval().requires_grad_(False)
    rm.zero_grad(set_to_none=True)


def soft_overlong_reward(length,max_new,soft_overlong=0,penalty=1.0):
    """Ramp from 0 to -penalty over the last soft_overlong response tokens.

    EOS counts as a generated token, including when it occurs at the limit.
    A zero buffer disables shaping (the one-token tiny demonstration default).
    """
    if not 0<=soft_overlong<=max_new or max_new<1:
        raise ValueError('soft_overlong must be between 0 and max_new')
    if not math.isfinite(penalty) or penalty<0:
        raise ValueError('overlong penalty must be finite and nonnegative')
    if soft_overlong==0:
        return 0.0
    excess=max(0,length-(max_new-soft_overlong))
    return -penalty*min(1.0,excess/soft_overlong)


# ================================================================
# 07. DAPO: DECOUPLED CLIP AND DYNAMIC SAMPLING; NO CRITIC.
# Preference labels train the RM, which scores fresh actor responses.
# Drop groups without reward variation and refill BEFORE updating.
# Average the clipped loss across ALL response tokens in the batch.
# ================================================================


@torch.no_grad()
def group_rollout(actor,rm,prompt,group_size,max_new,allowed,
                  soft_overlong=0,overlong_penalty=1.0,eos=7):
    """Cache old log probabilities and group-relative shaped RM rewards."""
    if group_size<2:
        raise ValueError('group_size must be at least 2')
    out=[]
    for _ in range(group_size):
        actions=sample_tokens(actor,prompt,max_new=max_new,eos=eos,allowed_ids=allowed)
        old_lp,_=positions_log_probs(actor,prompt,actions,allowed_ids=allowed)
        raw_reward=rm(prompt,actions).item()
        length_reward=soft_overlong_reward(len(actions),max_new,soft_overlong,overlong_penalty)
        out.append(dict(prompt_ids=prompt,actions=actions,old_lp=old_lp.detach(),
                        raw_reward=raw_reward,length_reward=length_reward,
                        reward=raw_reward+length_reward,
                        truncated=len(actions)==max_new and actions[-1]!=eos))
    rewards=torch.tensor([item['reward'] for item in out],device=out[0]['old_lp'].device)
    if not torch.isfinite(rewards).all():
        raise ValueError('Reward model must produce finite scores')
    advantages=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
    for item,advantage in zip(out,advantages):
        item['advantage']=advantage.detach()
    return out


def dynamic_rollout(actor,rm,records,group_size,max_new,allowed,
                    batch_size=1,max_attempts=16,start_index=0,min_reward_std=1e-6,
                    soft_overlong=0,overlong_penalty=1.0):
    """Cycle through prompts, rejecting groups with negligible reward spread.

    All groups come from the same unchanged actor. The attempt cap avoids an
    infinite loop for constant rewards or a collapsed policy. A partial batch
    is returned for diagnostics; callers must skip its optimizer update.
    """
    if not records or batch_size<1 or max_attempts<batch_size:
        raise ValueError('Need records, batch_size >= 1 and max_attempts >= batch_size')
    if not math.isfinite(min_reward_std) or min_reward_std<0:
        raise ValueError('min_reward_std must be finite and nonnegative')
    groups=[]
    for attempt in range(max_attempts):
        prompt=records[(start_index+attempt)%len(records)]['prompt_ids']
        group=group_rollout(actor,rm,prompt,group_size,max_new,allowed,
                            soft_overlong,overlong_penalty)
        rewards=torch.tensor([item['reward'] for item in group],device=group[0]['old_lp'].device)
        # For continuous preference scores, replace binary accuracy filtering
        # with a reward-spread test. Tied scores are excluded even at threshold 0.
        if rewards.std(unbiased=False).item()>min_reward_std:
            groups.append(group)
        if len(groups)==batch_size:
            return groups,attempt+1
    return groups,max_attempts


def dapo_update(actor,optimizer,groups,allowed,clip_low=0.2,clip_high=0.28,epochs=2):
    """Token-level clipped policy gradient, with no KL or value loss.

    L = -sum_(i,t) min(ratio*A_i, clip(ratio,1-low,1+high)*A_i)
        / sum_i len(response_i).

    The denominator covers every response token across every accepted group;
    it is NOT a mean of response means. Cached old logprobs and advantages
    remain fixed over update epochs. No reference policy is needed.
    """
    if (not math.isfinite(clip_low) or not math.isfinite(clip_high)
        or not 0<=clip_low<1 or clip_high<clip_low or epochs<1):
        raise ValueError('Need 0 <= clip_low < 1, clip_high >= clip_low and epochs >= 1')
    if not groups or any(not group or any(not item['actions'] for item in group)
                         for group in groups):
        raise ValueError('DAPO needs nonempty groups and responses')
    token_count=sum(len(item['actions']) for group in groups for item in group)
    for _ in range(epochs):
        losses=[]
        for group in groups:
            for item in group:
                new_lp,_=positions_log_probs(actor,item['prompt_ids'],item['actions'],
                                              allowed_ids=allowed)
                ratio=(new_lp-item['old_lp']).exp()
                advantage=item['advantage']
                clipped=ratio.clamp(1-clip_low,1+clip_high)
                losses.append(-torch.minimum(ratio*advantage,clipped*advantage).sum())
        loss=torch.stack(losses).sum()/token_count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step()
    return loss.item()


def main():
    a=argparse.ArgumentParser(description='LFM2.5-230M DAPO with learned preference rewards')
    a.add_argument('--checkpoint',help='Path to official model.safetensors')
    a.add_argument('--data',help='Tokenized prompt/chosen/rejected JSONL; same as script 06')
    a.add_argument('--tiny',action='store_true',help='Small RANDOM model, NOT pretrained')
    a.add_argument('--steps',type=int,default=2)
    a.add_argument('--max-new',type=int,default=16)
    a.add_argument('--lr',type=float,default=1e-5)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--group-size',type=int,default=4)
    a.add_argument('--rm-epochs',type=int,default=3)
    a.add_argument('--rm-lr',type=float,default=1e-3,help='Preference reward-head learning rate')
    a.add_argument('--clip-low',type=float,default=0.2)
    a.add_argument('--clip-high',type=float,default=0.28)
    a.add_argument('--epochs',type=int,default=2,help='Actor updates per fresh rollout batch')
    a.add_argument('--batch-size',type=int,default=1,help='Accepted prompt groups per update')
    a.add_argument('--max-sampling-attempts',type=int,default=16,
                   help='Maximum candidate groups per step; skip update if batch is not filled')
    a.add_argument('--min-reward-std',type=float,default=1e-6,
                   help='Reject preference-reward groups at or below this population std')
    a.add_argument('--soft-overlong',type=int,default=None,
                   help='Length-penalty buffer; default: last quarter of max-new (tiny: 0); 0 disables')
    a.add_argument('--overlong-penalty',type=float,default=1.0,
                   help='Maximum additive length penalty in reward-model score units')
    args=a.parse_args()
    if args.group_size<2 or args.rm_epochs<1 or args.max_new<1 or args.epochs<1 or args.steps<0:
        a.error('Need group-size >= 2, rm-epochs/max-new/epochs >= 1 and steps >= 0')
    if args.batch_size<1 or args.max_sampling_attempts<args.batch_size:
        a.error('Need batch-size >= 1 and max-sampling-attempts >= batch-size')
    if (not math.isfinite(args.clip_low) or not math.isfinite(args.clip_high)
        or not 0<=args.clip_low<1 or args.clip_high<args.clip_low):
        a.error('Need 0 <= clip-low < 1 and clip-high >= clip-low')
    if not math.isfinite(args.min_reward_std) or args.min_reward_std<0:
        a.error('min-reward-std must be finite and nonnegative')
    max_new=1 if args.tiny else args.max_new
    soft_overlong=args.soft_overlong
    if soft_overlong is None:
        soft_overlong=0 if args.tiny else max(1,max_new//4)
    soft_overlong_reward(0,max_new,soft_overlong,args.overlong_penalty)
    torch.manual_seed(args.seed)
    actor,_=initialize(args.tiny,args.checkpoint,critic=False)
    if args.data:
        prefs=load_jsonl(args.data,actor.model.embed_tokens.num_embeddings)
    elif args.tiny:
        prefs=[{'prompt_ids':[1,22,33],'chosen_ids':[10],'rejected_ids':[11]},
               {'prompt_ids':[1,34,35],'chosen_ids':[10],'rejected_ids':[12]}]
    else:
        raise ValueError('Supply --data preference.jsonl with prompt/chosen/rejected token ids')
    rm=LearnedReward(actor,cfg(args.tiny))
    train_reward_model(rm,prefs,epochs=args.rm_epochs,lr=args.rm_lr)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    allowed=[10,11,12,13] if args.tiny else None
    cursor=0
    for step in range(args.steps):
        groups,attempts=dynamic_rollout(
            actor,rm,prefs,args.group_size,max_new,allowed,
            batch_size=args.batch_size,max_attempts=args.max_sampling_attempts,
            start_index=cursor,min_reward_std=args.min_reward_std,
            soft_overlong=soft_overlong,overlong_penalty=args.overlong_penalty)
        cursor=(cursor+attempts)%len(prefs)
        if len(groups)<args.batch_size:
            print(f'DAPO preference step={step} skipped: accepted_groups={len(groups)}/'
                  f'{args.batch_size} sampled_groups={attempts}; no optimizer update')
            continue
        loss=dapo_update(actor,optimizer,groups,allowed,
                         clip_low=args.clip_low,clip_high=args.clip_high,epochs=args.epochs)
        items=[item for group in groups for item in group]
        print(f'DAPO preference step={step} loss={loss:.4f} '
              f'accepted_groups={len(groups)} sampled_groups={attempts} '
              f'tokens={sum(len(item["actions"]) for item in items)} '
              f'truncated={sum(item["truncated"] for item in items)}')
        print('    raw_rewards=',[round(item['raw_reward'],3) for item in items],
              'rewards=',[round(item['reward'],3) for item in items],
              'advantages=',[round(item['advantage'].item(),3) for item in items])
        print('    sampled_token_ids=',[item['actions'] for item in items])


if __name__=='__main__':
    main()
