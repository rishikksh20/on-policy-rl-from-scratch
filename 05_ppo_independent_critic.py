"""LFM2.5-230M PPO with an independent pretrained LFM critic.

Requires torch + Python standard library. Model code lives in lfm.py.
Both backbones load the same official model.safetensors into separate parameters.
The actor has a tied LM head; the critic has a newly initialized scalar head.
Reward and PPO equations follow 01_ppo_deterministic.py.

    python 05_ppo_independent_critic.py --tiny --steps 3 --lr 0.001
    python 05_ppo_independent_critic.py --checkpoint checkpoint/model.safetensors \
        --data tokenized/numeric_tokens.jsonl --steps 20 --critic-lr 1e-5
"""
import argparse
import copy
import json

import torch
from torch import nn
import torch.nn.functional as F

from lfm import initialize, initialize_critic


def load_jsonl(path):
    """JSON Lines with prompt_ids/answer_ids from the OFFICIAL LFM tokenizer."""
    with open(path,encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def positions_log_probs(policy,prompt,actions,allowed_ids=None):
    """Teacher-force sampled actions to recover log pi(a_t | s_t), shape [T]."""
    device=next(policy.parameters()).device
    ids=torch.tensor([prompt+actions[:-1]],dtype=torch.long,device=device)
    logits,_=policy(ids)
    offset=len(prompt)-1
    selection=logits[:,offset:offset+len(actions),:].float()
    if allowed_ids is not None:
        # Match the restricted sampling distribution in tiny mode.
        selection=selection[:,:,allowed_ids]
        mapped=[allowed_ids.index(a) for a in actions]
    else:
        mapped=actions
    act=torch.tensor(mapped,dtype=torch.long,device=device).view(1,-1,1)
    return F.log_softmax(selection,dim=-1).gather(-1,act).squeeze(-1).squeeze(0)


def positions_values(critic,prompt,actions):
    """Values BEFORE each sampled action, using the actor's exact prefixes.

    Input: prompt + actions[:-1]; take positions P-1 .. P+T-2.
    The causal critic cannot see the action it is about to evaluate or later tokens.
    """
    device=next(critic.parameters()).device
    ids=torch.tensor([prompt+actions[:-1]],dtype=torch.long,device=device)
    values=critic(ids)
    offset=len(prompt)-1
    return values[:,offset:offset+len(actions)].squeeze(0)


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
    """Objective verifier: exact token sequence after removing trailing EOS."""
    def clean(xs):
        xs=list(xs)
        return xs[:-1] if xs and xs[-1]==eos else xs
    return float(clean(generated)==clean(answer))


def show_rollout(step,prompt,completion,score,loss):
    print(f'step={step} prompt_ids={prompt[:5]} response_ids={completion[:8]} '
          f'reward={score:+.2f} loss={loss:.4f}')


def common_args(description):
    a=argparse.ArgumentParser(description=description)
    a.add_argument('--checkpoint',help='Official model.safetensors for BOTH backbones')
    a.add_argument('--data',help='Pre-tokenized JSONL with prompt_ids and answer_ids')
    a.add_argument('--tiny',action='store_true',help='Small RANDOM models for debugging, NOT 230M')
    a.add_argument('--steps',type=int,default=2)
    a.add_argument('--max-new',type=int,default=16)
    a.add_argument('--lr',type=float,default=1e-5,help='Actor learning rate')
    a.add_argument('--critic-lr',type=float,default=None,help='Critic learning rate; default: --lr')
    a.add_argument('--seed',type=int,default=42)
    return a


# ================================================================
# 05. PPO with an INDEPENDENT pretrained LFM critic.
# Actor: LFM backbone + tied vocabulary head -> next-token probabilities.
# Critic: separate LFM backbone + scalar head -> expected future return.
# Reference: frozen actor copy for the per-token KL reward penalty.
# GAE returns and advantages are frozen BEFORE either optimizer updates.
# ================================================================


def gae(rewards,old_values,bootstrap,gamma=0.99,lam=0.95):
    """Return[t] = Advantage[t] + OLD critic value[t]."""
    advantages=torch.zeros_like(rewards)
    acc=rewards.new_zeros(())
    next_v=bootstrap
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_v-old_values[t]
        acc=delta+gamma*lam*acc
        advantages[t]=acc
        next_v=old_values[t]
    return advantages,advantages+old_values


@torch.no_grad()
def rollout(actor,critic,reference,record,max_new,allowed,beta=0.02):
    """Cache actor logprobs and separate-critic GAE targets before any updates."""
    prompt=record['prompt_ids']
    actions=sample_tokens(actor,prompt,max_new=max_new,allowed_ids=allowed)
    old_lp=positions_log_probs(actor,prompt,actions,allowed_ids=allowed)
    ref_lp=positions_log_probs(reference,prompt,actions,allowed_ids=allowed)
    old_v=positions_values(critic,prompt,actions)
    score=token_reward(actions,record['answer_ids'])
    rewards=-beta*(old_lp-ref_lp)
    rewards[-1]+=score
    if actions[-1]==7: # EOS means terminal: no future value.
        bootstrap=old_v.new_zeros(())
    else:
        # Token limit is truncation; bootstrap from the INDEPENDENT critic.
        ids=torch.tensor([prompt+actions],dtype=torch.long,device=old_v.device)
        bootstrap=critic(ids)[0,-1]
    advantages,returns=gae(rewards,old_v,bootstrap)
    return dict(prompt=prompt,actions=actions,old_lp=old_lp.detach(),
                old_values=old_v.detach(),advantages=advantages.detach(),
                returns=returns.detach(),score=score,rewards=rewards.detach())


def ppo_update(actor,critic,actor_opt,critic_opt,batch,allowed,
               clip_eps=0.2,value_coef=0.5,epochs=2):
    """Clipped actor loss and separate-critic value regression.

    Actor gradients update ONLY the actor backbone and LM head.
    Critic gradients update ONLY the critic backbone and scalar value head.
    Both losses use fixed rollout advantages/returns throughout the PPO epochs.
    """
    for _ in range(epochs):
        lp=positions_log_probs(actor,batch['prompt'],batch['actions'],allowed_ids=allowed)
        ratio=(lp-batch['old_lp']).exp()
        advantage=batch['advantages']
        policy_loss=-torch.minimum(ratio*advantage,
            ratio.clamp(1-clip_eps,1+clip_eps)*advantage).mean()
        actor_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.0)
        actor_opt.step()

        values=positions_values(critic,batch['prompt'],batch['actions'])
        value_loss=.5*(values-batch['returns']).square().mean()
        critic_opt.zero_grad(set_to_none=True)
        (value_coef*value_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(),1.0)
        critic_opt.step()
    loss=policy_loss+value_coef*value_loss
    return loss.item(),policy_loss.item(),value_loss.item()


def main():
    args=common_args('Pure PyTorch LFM2.5-230M PPO with independent critic').parse_args()
    torch.manual_seed(args.seed)
    actor,device=initialize(args.tiny,args.checkpoint)
    critic,_=initialize_critic(args.tiny,args.checkpoint,device=device)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    actor_opt=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    critic_lr=args.lr if args.critic_lr is None else args.critic_lr
    critic_opt=torch.optim.AdamW(critic.parameters(),lr=critic_lr)
    if args.data:
        records=load_jsonl(args.data)
    elif args.tiny:
        records=[{'prompt_ids':[1,22,33],'answer_ids':[10]}]
    else:
        raise ValueError('Supply --data numeric.jsonl for real LFM training')
    allowed=[10,11,12,13] if args.tiny else None
    max_new=1 if args.tiny else args.max_new
    print('PPO independent critic | model:',
          'tiny random smoke test' if args.tiny else 'REAL LFM2.5-230M',
          f'| actor_lr={args.lr} critic_lr={critic_lr}')
    for step in range(args.steps):
        batch=rollout(actor,critic,reference,records[step%len(records)],max_new,allowed)
        total,policy,value=ppo_update(actor,critic,actor_opt,critic_opt,batch,allowed)
        show_rollout(step,batch['prompt'],batch['actions'],batch['score'],total)
        print('   reward_vector=',batch['rewards'].tolist(),
              'critic_targets=',batch['returns'].tolist(),
              'advantages=',batch['advantages'].tolist(),
              f'policy_loss={policy:.3f} value_loss={value:.3f}')


if __name__=='__main__':
    main()
