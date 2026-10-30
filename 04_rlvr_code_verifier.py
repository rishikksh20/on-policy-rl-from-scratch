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
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from lfm import initialize


def load_jsonl(path,vocab_size=65536):
    """JSON Lines: one dict per line, numeric token ids from OFFICIAL tokenizer."""
    with open(path,encoding='utf8') as f:
        records=[json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError('Data must contain at least one record')
    for record in records:
        for key in ('prompt_ids',):
            ids=record.get(key) if isinstance(record,dict) else None
            if (not isinstance(ids,list) or not ids
                or any(type(i) is not int or not 0<=i<vocab_size for i in ids)):
                raise ValueError(f'{key} must contain nonempty integer token IDs in [0, {vocab_size})')
        tests=record.get('tests')
        if (not isinstance(tests,list) or not tests
            or any(not isinstance(t,dict) or any(type(t.get(k)) is not int
                   for k in ('a','b','expected')) for t in tests)):
            raise ValueError('tests must contain integer a, b, and expected fields')
    return records


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
# 04. RLVR (Reinforcement Learning with VERIFIABLE Rewards)
# RLVR describes HOW the reward is computed, not a new optimizer.
# We use GRPO as optimizer here. Verifier: arithmetic code unit tests.
# - REAL: generate code tokens; decode using model's tokenizer.json.
# - TINY: four predefined expressions, encoded as single action IDs,
#         solely to test the loop without a tokenizer/checkpoint.
# Python AST whitelist interprets arithmetic; NEVER exec() generated code.
# ================================================================
import ast
import copy
import re


class Decoder:
    """Minimal ID-to-text decoder for BPE ByteLevel or SentencePiece metaspace.

    Decode ONLY (encoding must be done by the official LFM tokenizer).
    Supports common ByteLevel/Metaspace models; tokenizer.json may contain
    additional decoder stages beyond this educational subset. In that case
    verify decoded output against the official tokenizer before real training.
    """
    def __init__(self,filename):
        raw=json.loads(Path(filename).read_text(encoding='utf8'))
        model=raw['model']
        vocab=model['vocab']
        if isinstance(vocab,dict):
            self.ids={int(i):word for word,i in vocab.items()}
        else:
            self.ids={i:entry[0] for i,entry in enumerate(vocab)}
        self.special={int(entry['id']) for entry in raw.get('added_tokens',[])
                      if entry.get('special')}
        decoder=json.dumps(raw.get('decoder',{}))
        self.bytelevel='ByteLevel' in decoder
        self.metaspace='Metaspace' in decoder or '▁' in ' '.join(
            list(self.ids.values())[:300])
        # GPT2/HF ByteLevel decoder's reversible byte -> Unicode lookup
        # Reversible GPT-2 byte-to-Unicode mapping:
        standard=list(range(33,127))+list(range(161,173))+list(range(174,256))
        extra=[b for b in range(256) if b not in standard]
        self.inv={chr(c):b for b,c in zip(standard+extra,
                    standard+list(range(256,256+len(extra))))}

    def decode(self,ids):
        s=''.join(self.ids.get(int(i),'') for i in ids if int(i) not in self.special)
        if self.bytelevel:
            data=bytes(self.inv.get(ch,ord(ch) if ord(ch)<256 else 63) for ch in s)
            return data.decode('utf8',errors='replace')
        if self.metaspace:
            return s.replace('▁',' ')
        # ByteFallback <0xXX> tokens or other decoder chains aren't supported.
        return s


def arithmetic(node,a,b):
    """Interpret a tiny safe AST grammar, no calls, imports, loops or exec."""
    if isinstance(node,ast.Expression):
        return arithmetic(node.body,a,b)
    if isinstance(node,ast.Name) and node.id in {'a','b'}:
        return a if node.id=='a' else b
    if isinstance(node,ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
        return -arithmetic(node.operand,a,b)
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
        left=arithmetic(node.left,a,b)
        right=arithmetic(node.right,a,b)
        if isinstance(node.op,ast.Add): return left+right
        if isinstance(node.op,ast.Sub): return left-right
        return left*right
    raise ValueError('Unsafe or unsupported generated program')


def extract_expression(code):
    """Accept a bare expression or `def solve(a,b): return a+b`."""
    code=code.strip()
    if code.startswith('```'):
        code=re.sub(r'^```(?:python)?\s*','',code)
        code=re.sub(r'```\s*$','',code).strip()
    tree=ast.parse(code)
    if len(tree.body)!=1:
        raise ValueError('Expected one expression or one function')
    top=tree.body[0]
    if isinstance(top,ast.Expr): return ast.Expression(top.value)
    if (isinstance(top,ast.FunctionDef) and top.name=='solve'
        and not top.decorator_list and top.returns is None
        and top.type_comment is None and not getattr(top,'type_params',[])
        and len(top.args.args)==2
        and [p.arg for p in top.args.args]==['a','b']
        and all(p.annotation is None and p.type_comment is None for p in top.args.args)
        and not top.args.posonlyargs and not top.args.kwonlyargs
        and not top.args.defaults and not top.args.kw_defaults
        and top.args.vararg is None and top.args.kwarg is None
        and len(top.body)==1 and isinstance(top.body[0],ast.Return)):
        return ast.Expression(top.body[0].value)
    raise ValueError('Unrecognized code format')


def verify_unit_tests(code,tests):
    """Binary deterministic verifier: all unit tests must pass => +1.

    tests: [{"a":2,"b":3,"expected":5}, ...]
    Returns 0 for syntax errors, unsafe constructs or incorrect outputs.
    In a real coding RLVR environment, use a separately sandboxed
    subprocess/container plus timeouts, never unrestricted eval/exec.
    """
    try:
        expr=extract_expression(code)
        return float(bool(tests) and all(
            arithmetic(expr,int(t['a']),int(t['b']))==int(t['expected'])
            for t in tests))
    except (ValueError,SyntaxError,RecursionError,OverflowError,TypeError):
        return 0.0


@torch.no_grad()
def verified_group(actor,reference,record,group_size,max_new,allowed,decoder):
    """Generate G codes and score with deterministic unit tests."""
    prompt=record['prompt_ids']
    toy_map={10:'a+b',11:'a-b',12:'a*b',13:'a'}
    group=[]
    for _ in range(group_size):
        actions=sample_tokens(actor,prompt,max_new=max_new,allowed_ids=allowed)
        logp,_=positions_log_probs(actor,prompt,actions,allowed_ids=allowed)
        ref_lp,_=positions_log_probs(reference,prompt,actions,allowed_ids=allowed)
        code=toy_map[actions[0]] if allowed is not None else decoder.decode(actions)
        reward=verify_unit_tests(code,record['tests'])
        group.append(dict(actions=actions,code=code,reward=reward,
                          old_lp=logp.detach(),ref_lp=ref_lp.detach()))
    scores=torch.tensor([b['reward'] for b in group],device=group[0]['old_lp'].device)
    advantages=(scores-scores.mean())/(scores.std(unbiased=False)+1e-6)
    for sample,adv in zip(group,advantages):
        sample['adv']=adv.detach()
    return group


def update_with_grpo(actor,opt,prompt,group,allowed,beta=0.02,eps=0.2,epochs=2):
    """Identical GRPO math to script 03, with verified program rewards."""
    for _ in range(epochs):
        loss_terms=[]
        for item in group:
            new_lp,_=positions_log_probs(actor,prompt,item['actions'],allowed_ids=allowed)
            ratio=(new_lp-item['old_lp']).exp()
            a=item['adv']
            actor_loss=-torch.minimum(ratio*a,ratio.clamp(1-eps,1+eps)*a).mean()
            d=item['ref_lp']-new_lp
            kl=(d.exp()-d-1).mean()
            loss_terms.append(actor_loss+beta*kl)
        loss=torch.stack(loss_terms).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.)
        opt.step()
    return loss.item()


def main():
    ap=common_args('LFM2.5-230M PyTorch RLVR with a coding verifier')
    ap.add_argument('--tokenizer',help='Official local tokenizer.json, decoding only')
    ap.add_argument('--group-size',type=int,default=4)
    args=ap.parse_args()
    if args.max_new<1:
        raise ValueError('--max-new must be at least 1')
    if args.group_size<2:
        raise ValueError('--group-size must be at least 2 for group-relative advantages')
    torch.manual_seed(args.seed)
    actor,device=initialize(args.tiny,args.checkpoint)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    opt=torch.optim.AdamW(actor.parameters(),lr=args.lr)
    if args.data:
        data=load_jsonl(args.data,actor.model.embed_tokens.num_embeddings)
    elif args.tiny:
        data=[{'prompt_ids':[1,22,33],
               'tests':[{'a':2,'b':3,'expected':6},{'a':-4,'b':6,'expected':-24}]}]
    else:
        raise ValueError('Supply --data coding.jsonl')
    if not args.tiny and not args.tokenizer:
        raise ValueError('Supply --tokenizer /path/to/tokenizer.json for generated code decoding')
    decoder=None if args.tiny else Decoder(args.tokenizer)
    allowed=[10,11,12,13] if args.tiny else None
    max_new=1 if args.tiny else args.max_new
    for step in range(args.steps):
        rec=data[step%len(data)]
        group=verified_group(actor,reference,rec,args.group_size,max_new,allowed,decoder)
        loss=update_with_grpo(actor,opt,rec['prompt_ids'],group,allowed)
        print(f'RLVR step={step} loss={loss:.4f}')
        for item in group:
            print(f'   code={item["code"]!r} tests_pass={bool(item["reward"])} '
                  f'advantage={item["adv"].item():+.3f}')


if __name__=='__main__':
    main()
