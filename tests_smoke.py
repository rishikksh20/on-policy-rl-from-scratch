"""Small offline smoke tests for checkpoint decoding, equations, and verifier.
Run: python tests_smoke.py  (torch + stdlib only)
"""
import copy
import importlib.util
import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch

import lfm

HERE=Path(__file__).parent

def module(file,name):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ppo=module('01_ppo_deterministic.py','ppo')
ppo_preference=module('02_ppo_preference_reward.py','ppo_preference')
grpo=module('03_grpo.py','grpo')
rlvr=module('04_rlvr_code_verifier.py','rlvr')
independent=module('05_ppo_independent_critic.py','independent')
grpo_preference=module('06_grpo_preference_reward.py','grpo_preference')
dapo_preference=module('07_dapo_preference_reward.py','dapo_preference')


def write_safetensors(path,tensors):
    """Write tiny F32/BF16 fixtures without a safetensors package dependency."""
    h={}; parts=[]; offset=0
    for name,v in tensors.items():
        raw=v.contiguous().view(torch.uint8).numpy().tobytes()
        h[name]={'dtype':'BF16' if v.dtype==torch.bfloat16 else 'F32',
                 'shape':list(v.shape),'data_offsets':[offset,offset+len(raw)]}
        parts.append(raw); offset+=len(raw)
    header=json.dumps(h).encode('utf8')
    path.write_bytes(struct.pack('<Q',len(header))+header+b''.join(parts))

# 1. Exact architecture tensor count from public configuration (without allocation).
c=ppo.cfg()
assert len(c['layers'])==14
assert c['vocab']==65536 and c['dim']==1024
H,FF,V,NH,NK=c['dim'],c['ff'],c['vocab'],c['heads'],c['kv_heads']
N=V*H +8*(3*H*FF+2*H+4*H*H+H*3) +6*(3*H*FF+2*H+2*H*H+2*H*NK*(H//NH)+2*(H//NH))+H
assert N==229693184,N

# 2. Verify GAE exact returns and gradients on a small toy actor.
adv,ret=ppo.gae(torch.tensor([0.,1.]),torch.tensor([0.2,0.5]),torch.tensor(0.),gamma=1.0,lam=1.)
assert torch.allclose(adv,torch.tensor([0.8,0.5]),atol=1e-6)
assert torch.allclose(ret,torch.tensor([1.,1.]),atol=1e-6)

# 3. LFM forward output shape, critic output and causal token shift.
m=ppo.LFMPolicy(ppo.cfg(tiny=True),critic=True)
prompt=[1,22,33]; actions=[10,11]
logp,val=ppo.positions_log_probs(m,prompt,actions,critic=True)
assert logp.shape==val.shape==(2,)
logp.sum().backward()
assert m.model.embed_tokens.weight.grad is not None

# 4. Minimal safetensors roundtrip (F32 and BF16 test), no package dependency.
with tempfile.TemporaryDirectory() as tmp:
    tensors={'float32':torch.tensor([[1.,2.]],dtype=torch.float32),
             'bfloat16':torch.tensor([1.,2.,3.],dtype=torch.bfloat16)}
    path=Path(tmp)/'weights.safetensors'
    write_safetensors(path,tensors)
    result=ppo.read_safetensors(path)
    for key in tensors:
        assert torch.equal(result[key],tensors[key])

# 5. Verifier is functional and safe: does not execute arbitrary code.
tests=[{'a':2,'b':3,'expected':5},{'a':-4,'b':6,'expected':2}]
assert rlvr.verify_unit_tests('a+b',tests)==1.0
assert rlvr.verify_unit_tests('def solve(a,b):\n    return a+b',tests)==1.0
assert rlvr.verify_unit_tests('a*b',tests)==0.0
assert rlvr.verify_unit_tests('__import__("os").system("whoami")',tests)==0.0

# 6. One tiny pretrained checkpoint loads into disjoint policy/value backbones.
# Test tied checkpoints both with and without the redundant LM-head tensor.
torch.manual_seed(42)
pretrained,_=lfm.initialize(tiny=True,device='cpu')
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'model.safetensors'
    for include_lm_head in (False,True):
        state={n:v for n,v in pretrained.state_dict().items()
               if include_lm_head or n!='lm_head.weight'}
        write_safetensors(path,state)
        actor,_=lfm.initialize(tiny=True,ckpt=path,device='cpu')
        critic,_=lfm.initialize_critic(tiny=True,ckpt=path,device='cpu')
        for name,value in actor.model.state_dict().items():
            assert torch.equal(value,critic.model.state_dict()[name]),name
        # Loading the backbone must leave the added value head untouched.
        head_before={n:v.clone() for n,v in critic.value_head.state_dict().items()}
        lfm.load_weights(critic,path)
        for name,value in critic.value_head.state_dict().items():
            assert torch.equal(value,head_before[name]),name
    # A missing backbone weight must still be rejected for value-only models.
    state.pop('model.layers.0.conv.in_proj.weight')
    write_safetensors(path,state)
    try:
        lfm.load_weights(critic,path)
    except RuntimeError as exc:
        assert 'model.layers.0.conv.in_proj.weight' in str(exc)
    else:
        raise AssertionError('Incomplete critic backbone checkpoint was accepted')

assert not hasattr(actor,'value_head')
assert not hasattr(critic,'lm_head')
assert actor.lm_head.weight is actor.model.embed_tokens.weight
assert {p.data_ptr() for p in actor.parameters()}.isdisjoint(
    {p.data_ptr() for p in critic.parameters()})
assert all(p.requires_grad for p in critic.parameters())

# 7. Batched causal values equal separate evaluations of every pre-action prefix.
actions=[10,11,7]
with torch.no_grad():
    values=independent.positions_values(critic,prompt,actions)
    expected=torch.stack([critic(torch.tensor([prompt+actions[:t]]))[0,-1]
                          for t in range(len(actions))])
assert values.shape==(3,)
assert torch.allclose(values,expected,atol=1e-6)

# 8. Actor and critic losses cannot send gradients into the other network.
independent.positions_log_probs(actor,prompt,actions).sum().backward()
assert actor.model.embed_tokens.weight.grad is not None
assert all(p.grad is None for p in critic.parameters())
actor.zero_grad(set_to_none=True)
independent.positions_values(critic,prompt,actions).square().mean().backward()
assert critic.model.embed_tokens.weight.grad.abs().sum()>0
assert critic.value_head.weight.grad.abs().sum()>0
assert all(p.grad is None for p in actor.parameters())
critic.zero_grad(set_to_none=True)

# 9. EOS and token limit both finish the scored-answer task: zero bootstrap.
reference=copy.deepcopy(actor).eval().requires_grad_(False)
record={'prompt_ids':prompt,'answer_ids':[10]}
allowed=[7,10,11,12]
with patch.object(independent,'sample_tokens',return_value=[10,7]):
    batch=independent.rollout(actor,critic,reference,record,2,allowed)
assert batch['score']==1.0
assert torch.allclose(batch['returns'][-1],torch.tensor(1.0),atol=1e-6)
with patch.object(independent,'sample_tokens',return_value=[10,11]):
    truncated=independent.rollout(actor,critic,reference,record,2,allowed)
assert torch.allclose(truncated['returns'][-1],
                      truncated['rewards'][-1],atol=1e-6)
for result in (batch,truncated):
    for key in ('old_lp','old_values','advantages','returns','rewards'):
        assert result[key].shape==(2,)
        assert not result[key].requires_grad

# 10. PPO updates BOTH independent backbones + the value head, keeps reference
# fixed, and reuses the unchanged rollout targets over multiple PPO epochs.
actor_before={n:p.detach().clone() for n,p in actor.named_parameters()}
critic_before={n:p.detach().clone() for n,p in critic.named_parameters()}
reference_before={n:p.detach().clone() for n,p in reference.named_parameters()}
cached={key:batch[key].clone()
        for key in ('old_lp','old_values','advantages','returns','rewards')}
actor_opt=torch.optim.AdamW(actor.parameters(),lr=1e-3)
critic_opt=torch.optim.AdamW(critic.parameters(),lr=1e-3)
losses=independent.ppo_update(actor,critic,actor_opt,critic_opt,batch,allowed)
assert all(torch.isfinite(torch.tensor(value)) for value in losses)
assert any(not torch.equal(p,actor_before[n])
           for n,p in actor.model.named_parameters(prefix='model'))
assert any(not torch.equal(p,critic_before[n])
           for n,p in critic.model.named_parameters(prefix='model'))
assert not torch.equal(critic.value_head.weight,critic_before['value_head.weight'])
for name,p in reference.named_parameters():
    assert torch.equal(p,reference_before[name]),name
    assert p.grad is None
for key,value in cached.items():
    assert torch.equal(value,batch[key]),key

# 11. Preference supervision trains only the reward head, then freezes the RM.
torch.manual_seed(42)
actor,_=lfm.initialize(tiny=True,device='cpu')
reference=copy.deepcopy(actor).eval().requires_grad_(False)
actor_before={n:p.detach().clone() for n,p in actor.named_parameters()}
rm=grpo_preference.LearnedReward(actor,lfm.cfg(tiny=True))
rm_backbone_before={n:p.detach().clone() for n,p in rm.backbone.named_parameters()}
prefs=[{'prompt_ids':prompt,'chosen_ids':[10],'rejected_ids':[11]}]
with torch.no_grad():
    loss_before=torch.nn.functional.softplus(rm(prompt,[11])-rm(prompt,[10]))
grpo_preference.train_reward_model(rm,prefs,epochs=3,lr=0.01)
with torch.no_grad():
    loss_after=torch.nn.functional.softplus(rm(prompt,[11])-rm(prompt,[10]))
assert loss_after<loss_before
assert rm(prompt,[10]).item()>rm(prompt,[11]).item()
assert not rm.training and all(not p.requires_grad for p in rm.parameters())
assert not hasattr(actor,'value_head')
for name,p in actor.named_parameters():
    assert torch.equal(p,actor_before[name]),name
for name,p in rm.backbone.named_parameters():
    assert torch.equal(p,rm_backbone_before[name]),name

# 12. Group rewards come from the trained RM; tied scores produce zero advantages.
allowed=[10,11,12,13]
with patch.object(grpo_preference,'sample_tokens',side_effect=[[10],[11],[10],[11]]):
    group=grpo_preference.group_rollout(actor,reference,rm,prompt,4,1,allowed)
assert group[0]['reward']==rm(prompt,[10]).item()
assert group[1]['reward']==rm(prompt,[11]).item()
assert group[0]['advantage']>0 and group[1]['advantage']<0
for item in group:
    assert not item['old_lp'].requires_grad
    assert not item['ref_lp'].requires_grad
    assert not item['advantage'].requires_grad
with patch.object(grpo_preference,'sample_tokens',return_value=[10]):
    tied=grpo_preference.group_rollout(actor,reference,rm,prompt,4,1,allowed)
assert all(item['advantage'].item()==0.0 for item in tied)

# 13. GRPO increases preference for the higher-reward token and updates only actor.
rm.zero_grad(set_to_none=True)
rm_before={n:p.detach().clone() for n,p in rm.named_parameters()}
reference_before={n:p.detach().clone() for n,p in reference.named_parameters()}
with torch.no_grad():
    chosen,_=grpo_preference.positions_log_probs(actor,prompt,[10],allowed_ids=allowed)
    rejected,_=grpo_preference.positions_log_probs(actor,prompt,[11],allowed_ids=allowed)
    log_odds_before=chosen-rejected
optimizer=torch.optim.AdamW(actor.parameters(),lr=1e-3)
loss=grpo_preference.grpo_update(actor,optimizer,prompt,group,allowed,beta=0.,epochs=1)
assert torch.isfinite(torch.tensor(loss))
with torch.no_grad():
    chosen,_=grpo_preference.positions_log_probs(actor,prompt,[10],allowed_ids=allowed)
    rejected,_=grpo_preference.positions_log_probs(actor,prompt,[11],allowed_ids=allowed)
assert (chosen-rejected)>log_odds_before
for model,before in ((rm,rm_before),(reference,reference_before)):
    for name,p in model.named_parameters():
        assert torch.equal(p,before[name]),name
        assert p.grad is None
# 14. Reject function metadata outside the verifier's plain two-argument grammar.
for source in ('@unsupported()\ndef solve(a,b):\n    return a+b',
               'def solve(a,b=0):\n    return a+b',
               'def solve(a,b,*,c=0):\n    return a+b',
               'def solve(a,b,*args):\n    return a+b',
               'def solve(a,b,**kwargs):\n    return a+b',
               'def solve(a:int,b):\n    return a+b',
               'def solve(a,b)->int:\n    return a+b',
               'def solve(a,/,b):\n    return a+b'):
    assert rlvr.verify_unit_tests(source,tests)==0.0,source

# 15. Centered preference rewards are invariant to a shared raw-score offset.
torch.manual_seed(42)
ppo_actor,_=lfm.initialize(tiny=True,critic=True,device='cpu')
ppo_reference=copy.deepcopy(ppo_actor).eval().requires_grad_(False)
ppo_rm=ppo_preference.LearnedReward(ppo_actor,lfm.cfg(tiny=True))
ppo_preference.train_reward_model(ppo_rm,prefs,epochs=1)
with torch.no_grad():
    expected_center=torch.stack([ppo_rm(prompt,[10]),ppo_rm(prompt,[11])]).mean()
    assert torch.allclose(ppo_rm.score_center,expected_center,atol=1e-6)
    before=torch.tanh(ppo_rm(prompt,[10])-ppo_rm.score_center)
    ppo_rm.head.bias.add_(10.)
    ppo_rm.score_center.add_(10.)
    after=torch.tanh(ppo_rm(prompt,[10])-ppo_rm.score_center)
assert torch.allclose(before,after,atol=1e-6)
ppo_allowed=[7,10,11,12]
for actions in ([10,7],[10]):
    with patch.object(ppo,'sample_tokens',return_value=actions):
        result=ppo.rollout(ppo_actor,ppo_reference,record,len(actions),ppo_allowed)
    assert torch.allclose(result['returns'][-1],result['rewards'][-1],atol=1e-6)
    with patch.object(ppo_preference,'sample_tokens',return_value=actions):
        result=ppo_preference.rollout(ppo_actor,ppo_reference,ppo_rm,prompt,len(actions),ppo_allowed)
    assert torch.allclose(result['returns'][-1],torch.tensor(result['score']),atol=1e-6)

# 16. Sampling cannot silently disagree with rescoring or emit empty trajectories.
for script in (ppo,ppo_preference,grpo,rlvr,independent,grpo_preference,dapo_preference):
    for kwargs in ({'temperature':0.5},{'max_new':0},{'prompt':[]}):
        try:
            script.sample_tokens(actor,**({'prompt':prompt}|kwargs))
        except ValueError:
            pass
        else:
            raise AssertionError(f'{script.__name__} accepted invalid sampling options')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'data.jsonl'
        valid={'prompt_ids':prompt,'answer_ids':[10],'chosen_ids':[10],
               'rejected_ids':[11],'tests':tests}
        path.write_text(json.dumps(valid)+'\n')
        assert script.load_jsonl(path,vocab_size=96)==[valid]
        for row in (None,{}, {'prompt_ids':[]}, {'prompt_ids':[96]}, {'prompt_ids':[True]}):
            path.write_text('' if row is None else json.dumps(row)+'\n')
            try:
                script.load_jsonl(path,vocab_size=96)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{script.__name__} accepted invalid data: {row}')

# 17. DAPO uses the same pairwise RM training and keeps its backbone/actor fixed.
torch.manual_seed(42)
dapo=dapo_preference
dapo_actor,_=lfm.initialize(tiny=True,device='cpu')
dapo_rm=dapo.LearnedReward(dapo_actor,lfm.cfg(tiny=True))
actor_before={n:p.detach().clone() for n,p in dapo_actor.named_parameters()}
backbone_before={n:p.detach().clone() for n,p in dapo_rm.backbone.named_parameters()}
with torch.no_grad():
    margin_before=dapo_rm(prompt,[10])-dapo_rm(prompt,[11])
dapo.train_reward_model(dapo_rm,prefs,epochs=3,lr=0.01)
with torch.no_grad():
    assert dapo_rm(prompt,[10])-dapo_rm(prompt,[11])>margin_before
assert not dapo_rm.training and all(not p.requires_grad for p in dapo_rm.parameters())
assert all(p.grad is None for p in dapo_rm.parameters())
assert not hasattr(dapo_actor,'value_head')
for name,p in dapo_actor.named_parameters():
    assert torch.equal(p,actor_before[name]),name
for name,p in dapo_rm.backbone.named_parameters():
    assert torch.equal(p,backbone_before[name]),name

# 18. Soft length shaping is continuous at the buffer boundary and capped.
assert dapo.soft_overlong_reward(6,10,4)==0.0
assert dapo.soft_overlong_reward(7,10,4)==-0.25
assert dapo.soft_overlong_reward(10,10,4)==-1.0
assert dapo.soft_overlong_reward(12,10,4,penalty=2.)==-2.0
assert dapo.soft_overlong_reward(10,10,0)==0.0
for kwargs in ({'soft_overlong':11},{'soft_overlong':-1},{'penalty':-1.}):
    try:
        dapo.soft_overlong_reward(5,10,**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid soft-overlong options accepted')
with patch.object(dapo,'sample_tokens',side_effect=[[10,7],[11,12,7],[11,12,13]]):
    shaped=dapo.group_rollout(dapo_actor,dapo_rm,prompt,3,3,[7,10,11,12,13],soft_overlong=1)
assert [item['length_reward'] for item in shaped]==[0.,-1.,-1.]
assert [item['truncated'] for item in shaped]==[False,False,True]
for item in shaped:
    assert item['raw_reward']==dapo_rm(prompt,item['actions']).item()
    assert item['reward']==item['raw_reward']+item['length_reward']
    assert item['old_lp'].shape==(len(item['actions']),)
shaped_rewards=torch.tensor([item['reward'] for item in shaped])
assert torch.allclose(torch.stack([item['advantage'] for item in shaped]),
                      (shaped_rewards-shaped_rewards.mean())/(shaped_rewards.std(unbiased=False)+1e-6))

# 19. Dynamic sampling drops tied groups, advances prompts and fills a batch
# under one unchanged actor. Every accepted group has its OWN reward baseline.
dapo_records=prefs+[{'prompt_ids':[1,34,35],'chosen_ids':[10],'rejected_ids':[11]}]
with patch.object(dapo,'sample_tokens',
                  side_effect=[[10]]*4+[[10],[11],[10],[11]]*2) as sample:
    dapo_groups,attempts=dapo.dynamic_rollout(
        dapo_actor,dapo_rm,dapo_records,4,1,allowed,batch_size=2,max_attempts=3)
assert attempts==3 and len(dapo_groups)==2 and sample.call_count==12
assert dapo_groups[0][0]['prompt_ids']==dapo_records[1]['prompt_ids']
assert dapo_groups[1][0]['prompt_ids']==dapo_records[0]['prompt_ids']
for group in dapo_groups:
    rewards=torch.tensor([item['reward'] for item in group])
    advantages=torch.stack([item['advantage'] for item in group])
    assert torch.allclose(advantages,(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6))
    for item in group:
        assert item['reward']==dapo_rm(item['prompt_ids'],item['actions']).item()
        assert not item['old_lp'].requires_grad and not item['advantage'].requires_grad
        assert 'ref_lp' not in item
for name,p in dapo_actor.named_parameters():
    assert torch.equal(p,actor_before[name]),name
with patch.object(dapo,'sample_tokens',return_value=[10]) as sample:
    empty,attempts=dapo.dynamic_rollout(dapo_actor,dapo_rm,prefs,4,1,allowed,
                                       max_attempts=3,min_reward_std=0.)
assert empty==[] and attempts==3 and sample.call_count==12

# 20. Independent numeric example: asymmetric clipping AND global token mean.
# +A: ratio 1.25 still learns, 1.4 clips to 1.28. -A: 0.7 clips to 0.8;
# 0.85 and 1.5 still learn. Unequal lengths distinguish this from GRPO.
toy=torch.nn.Linear(1,1,bias=False)
torch.nn.init.zeros_(toy.weight)
positive=dict(prompt_ids=prompt,actions=[10,10],old_lp=torch.zeros(2),advantage=torch.tensor(1.))
negative=dict(prompt_ids=prompt,actions=[11,11,11],old_lp=torch.zeros(3),advantage=torch.tensor(-1.))
def toy_log_probs(policy,prompt,actions,allowed_ids=None):
    ratios=[1.25,1.4] if actions[0]==10 else [0.7,0.85,1.5]
    return torch.tensor(ratios).log()+policy.weight.sum(),None
with patch.object(dapo,'positions_log_probs',side_effect=toy_log_probs):
    loss=dapo.dapo_update(toy,torch.optim.SGD(toy.parameters(),lr=0.),
                          [[positive],[negative]],None,epochs=1)
assert abs(loss-(-1.25-1.28+0.8+0.85+1.5)/5)<1e-6
assert torch.allclose(toy.weight.grad,torch.tensor([[(-1.25+0.85+1.5)/5]]),atol=1e-6)

# 21. DAPO improves preference log odds using multiple groups/epochs, updates
# only the actor, and preserves cached behavior logprobs and advantages.
rm_before={n:p.detach().clone() for n,p in dapo_rm.named_parameters()}
cached=[(item['old_lp'].clone(),item['advantage'].clone())
        for group in dapo_groups for item in group]
def preference_log_odds():
    return torch.stack([
        dapo.positions_log_probs(dapo_actor,row['prompt_ids'],[10],allowed_ids=allowed)[0]
        -dapo.positions_log_probs(dapo_actor,row['prompt_ids'],[11],allowed_ids=allowed)[0]
        for row in dapo_records]).mean()
with torch.no_grad():
    odds_before=preference_log_odds()
loss=dapo.dapo_update(dapo_actor,torch.optim.AdamW(dapo_actor.parameters(),lr=1e-3),
                      dapo_groups,allowed,epochs=2)
assert torch.isfinite(torch.tensor(loss))
with torch.no_grad():
    assert preference_log_odds()>odds_before
assert any(not torch.equal(p,actor_before[n]) for n,p in dapo_actor.named_parameters())
for name,p in dapo_rm.named_parameters():
    assert torch.equal(p,rm_before[name]),name
    assert p.grad is None
for item,(old_lp,advantage) in zip((i for g in dapo_groups for i in g),cached):
    assert torch.equal(item['old_lp'],old_lp)
    assert torch.equal(item['advantage'],advantage)

# 22. Main skips an unfilled batch: no policy optimizer step, even weight decay.
actor_before={n:p.detach().clone() for n,p in dapo_actor.named_parameters()}
with patch('sys.argv',['07_dapo_preference_reward.py','--tiny','--steps','1',
                       '--batch-size','2','--max-sampling-attempts','2','--rm-epochs','1']), \
     patch.object(dapo,'initialize',return_value=(dapo_actor,'cpu')), \
     patch.object(dapo,'sample_tokens',side_effect=[[10],[11],[10],[11]]+[[10]]*4), \
     patch.object(dapo,'dapo_update') as update:
    dapo.main()
    update.assert_not_called()
for name,p in dapo_actor.named_parameters():
    assert torch.equal(p,actor_before[name]),name

print('All architecture, PPO/GRPO/DAPO, preference reward, independent critic, safetensors, verifier, and audit regression tests PASSED.')
