"""OPTIONAL one-time TEXT -> OFFICIAL LFM2.5 token IDs preprocessing.

This is the ONLY file that uses Transformers, because the checkpoint's
real tokenizer/chat template is not implemented by PyTorch itself.

    pip install transformers>=5.2
    python prepare_data_optional.py

All training scripts run on torch + stdlib ONLY and DO NOT import this.
You may instead supply already-tokenized JSONL from any tokenizer service.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL='LiquidAI/LFM2.5-230M'
tok=AutoTokenizer.from_pretrained(MODEL)
out=Path(__file__).parent/'tokenized'
out.mkdir(exist_ok=True)


def prompt_ids(question):
    """Apply the MODEL'S chat template and begin an assistant answer."""
    ids=tok.apply_chat_template(
        [{'role':'user','content':question}],
        tokenize=True,add_generation_prompt=True,
    )
    # Some tokenizer versions return a dict when return_dict=True; not used.
    return list(ids)


def response_ids(answer):
    """Encode target continuation (no second BOS) and one EOS token."""
    ids=tok.encode(answer,add_special_tokens=False)
    if not ids or ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    return ids


def write_jsonl(name,rows):
    path=out/name
    with path.open('w',encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(f'wrote {path} ({len(rows)} examples)')


numeric=[
    {'prompt':'What is 2 + 3? Answer with only the number.','answer':'5'},
    {'prompt':'What is 4 + 4? Answer with only the number.','answer':'8'},
    {'prompt':'What is 9 - 2? Answer with only the number.','answer':'7'},
]
write_jsonl('numeric_tokens.jsonl',[
    {'prompt_ids':prompt_ids(r['prompt']),'answer_ids':response_ids(r['answer'])}
    for r in numeric
])

preferences=[
    {'prompt':'A customer says their parcel is late. Reply politely and offer help.',
     'chosen':'I am sorry about the delay. Let me check the tracking and help with next steps.',
     'rejected':'Stop complaining. It is not my problem.'},
    {'prompt':'Explain why we should validate a machine-learning model on held-out data.',
     'chosen':'Held-out data helps estimate generalization and detect overfitting.',
     'rejected':'Validation is unnecessary because the training score proves the model works.'},
]
write_jsonl('preferences_tokens.jsonl',[
    {'prompt_ids':prompt_ids(r['prompt']),
     'chosen_ids':response_ids(r['chosen']),
     'rejected_ids':response_ids(r['rejected'])}
    for r in preferences
])

coding=[
    {'prompt':'Write ONLY a Python expression using variables a and b that adds the two integers. No markdown, no explanation.',
     'tests':[{'a':2,'b':3,'expected':5}, {'a':-4,'b':6,'expected':2}]},
    {'prompt':'Write ONLY a Python expression using variables a and b that multiplies two integers. No markdown, no explanation.',
     'tests':[{'a':2,'b':3,'expected':6}, {'a':-4,'b':6,'expected':-24}]},
]
write_jsonl('coding_tokens.jsonl',[
    {'prompt_ids':prompt_ids(r['prompt']),'tests':r['tests']}
    for r in coding
])

print('Decode sanity:',tok.decode(response_ids('a + b'),skip_special_tokens=True))
print('IMPORTANT: coding RLVR has a deliberately small AST grammar and '
      'will give zero reward to other equivalent-but-unsupported code.')
