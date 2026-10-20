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
grpo=module('03_grpo.py','grpo')
rlvr=module('04_rlvr_code_verifier.py','rlvr')
independent=module('05_ppo_independent_critic.py','independent')
grpo_preference=module('06_grpo_preference_reward.py','grpo_preference')


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

# 9. EOS has zero bootstrap; truncation uses the separate critic's end-state value.
reference=copy.deepcopy(actor).eval().requires_grad_(False)
record={'prompt_ids':prompt,'answer_ids':[10]}
allowed=[7,10,11,12]
with patch.object(independent,'sample_tokens',return_value=[10,7]):
    batch=independent.rollout(actor,critic,reference,record,2,allowed)
assert batch['score']==1.0
assert torch.allclose(batch['returns'][-1],torch.tensor(1.0),atol=1e-6)
with patch.object(independent,'sample_tokens',return_value=[10,11]):
    truncated=independent.rollout(actor,critic,reference,record,2,allowed)
with torch.no_grad():
    bootstrap=critic(torch.tensor([prompt+[10,11]]))[0,-1]
assert torch.allclose(truncated['returns'][-1],
                      truncated['rewards'][-1]+0.99*bootstrap,atol=1e-6)
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
print('All architecture, PPO/GRPO, preference reward, independent critic, safetensors, and verifier smoke tests PASSED.')
