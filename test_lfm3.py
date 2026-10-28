"""LFM3 offline checks and real-checkpoint generation in one file.

    pip install torch regex jinja2
    python test_lfm3.py --unit       # quick offline checks; only torch needed
    python test_lfm3.py              # checks, download, and generate
    python test_lfm3.py --offline    # checks and generate from local files

Generation defaults match the supplied pipeline example. Edit SAMPLING below
for advanced settings. Device and dtype are automatic: CUDA/BF16 or CPU/FP32.
No Hugging Face runtime libraries are imported. Generation additionally uses
regex for Unicode tokenization and Jinja2 for the official chat template.
"""
import argparse
import contextlib
import copy
import functools
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.request
from urllib.parse import quote

import torch
import torch.nn.functional as F

import lfm3


# Configuration: shared by the generation runner and command-line defaults.
MODEL_ID = 'LiquidAI/LFM3-1.7B'
DEFAULT_DIRECTORY = Path(__file__).resolve().parent / 'checkpoints' / 'LFM3-1.7B'
PROMPT = 'What is C. elegans?'
SAMPLING = dict(max_new_tokens=2048, temperature=0.1, top_k=50,
                repetition_penalty=1.1, seed=42)


# Checkpoint downloads and architecture validation.


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _request(url, **headers):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'lfm3-pytorch-local-test/1.0', **headers}), timeout=120)


def _valid_file(path, info):
    if not path.is_file() or path.stat().st_size != info['size']:
        return False
    lfs = info.get('lfs', {})
    hasher = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        hasher.update(f'blob {info["size"]}\0'.encode())
    with path.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest() == (lfs['sha256'] if lfs else info['blobId'])


def _download(root, revision, name, info):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError(f'Invalid checkpoint filename: {name}')
    if _valid_file(path, info):
        print(f'Using verified local {name}', file=sys.stderr, flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    start = partial.stat().st_size if partial.exists() else 0
    if start >= info['size']:
        # A completed partial file can be promoted without another network request.
        if _valid_file(partial, info):
            partial.replace(path)
            return path
        start = 0
    url = f'https://huggingface.co/{MODEL_ID}/resolve/{quote(revision, safe="")}/{quote(name)}'
    headers = {'Range': f'bytes={start}-'} if start else {}
    print(f'Downloading {name} ({info["size"] / 1e6:.1f} MB)', file=sys.stderr, flush=True)
    with _request(url, **headers) as response:
        append = start > 0 and response.status == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
            raise RuntimeError(f'Invalid HTTP resume offset for {name}')
        written = start if append else 0
        last = time.monotonic()
        with partial.open('ab' if append else 'wb') as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                written += len(chunk)
                if time.monotonic() - last >= 10:
                    print(f'  {name}: {written / 1e6:.0f}/{info["size"] / 1e6:.0f} MB',
                          file=sys.stderr, flush=True)
                    last = time.monotonic()
    if not _valid_file(partial, info):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'Checksum/size verification failed for {name}; retry the download')
    partial.replace(path)
    return path


def prepare_checkpoint(directory=DEFAULT_DIRECTORY, revision='main', offline=False):
    """Pin one repository revision and verify downloaded files before using them."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'download_manifest.json'
    if offline:
        if not manifest.is_file():
            raise FileNotFoundError('Offline mode requires a previous verified download_manifest.json')
        metadata = read_json(manifest)
        if metadata['model_id'] != MODEL_ID:
            raise ValueError('Local checkpoint manifest belongs to another model')
        for name, info in metadata['files'].items():
            if not _valid_file(root / name, info):
                raise RuntimeError(f'Offline checkpoint file is missing or corrupt: {name}')
        return root, metadata['revision']
    api = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{quote(revision, safe="")}?blobs=true'
    with _request(api) as response:
        metadata = json.load(response)
    revision = metadata['sha']
    files = {info['rfilename']: info for info in metadata['siblings']}
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
             'generation_config.json', 'chat_template.jinja', 'LICENSE']
    if 'special_tokens_map.json' in files:
        names.append('special_tokens_map.json')
    if 'model.safetensors.index.json' in files:
        name = 'model.safetensors.index.json'
        index = read_json(_download(root, revision, name, files[name]))
        names += [name, *dict.fromkeys(index['weight_map'].values())]
    else:
        names.append('model.safetensors')
    for name in names:
        if name not in files:
            raise ValueError(f'{MODEL_ID} does not provide required file {name}')
        _download(root, revision, name, files[name])
    record = dict(model_id=MODEL_ID, revision=revision, files={name: files[name] for name in names})
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return root, revision


def validate_model_config(raw):
    """Fail before loading if the hosted architecture differs from lfm3.cfg()."""
    c = lfm3.cfg()
    expected = dict(model_type='lfm3', vocab_size=c['vocab'], hidden_size=c['dim'],
                    intermediate_size=c['ff'], num_hidden_layers=len(c['layers']),
                    num_attention_heads=c['heads'], num_key_value_heads=c['kv_heads'],
                    head_dim=c['head_dim'], layer_types=c['layers'], conv_L_cache=c['conv_kernel'],
                    conv_bias=c['conv_bias'], norm_eps=c['norm_eps'],
                    use_sandwich_norm=c['use_sandwich_norm'], sliding_window=c['sliding_window'],
                    tie_word_embeddings=c['tie_word_embeddings'], rope_parameters=c['rope_parameters'],
                    pad_token_id=c['pad_token_id'], bos_token_id=c['bos_token_id'],
                    eos_token_id=c['eos_token_id'], max_position_embeddings=c['max_position_embeddings'])
    mismatch = {k: (raw.get(k), value) for k, value in expected.items() if raw.get(k) != value}
    if mismatch:
        raise ValueError(f'Downloaded model config differs from the local implementation: {mismatch}')


# Tokenization and the official chat template.


class LocalTokenizer:
    """ByteLevel BPE from tokenizer.json, including ignore_merges and added tokens.

    Implements the downloaded tokenizer's validated subset. No Hugging Face
    runtime is used; unsupported normalizers/pretokenizers fail explicitly.
    """
    def __init__(self, filename):
        try:
            import regex
        except ImportError as exc:
            raise RuntimeError('Install the Unicode regex dependency: pip install regex') from exc
        raw = read_json(filename)
        model = raw['model']
        pre = raw['pre_tokenizer']
        stages = pre.get('pretokenizers', [])
        if (model['type'] != 'BPE' or model.get('dropout') is not None
                or model.get('byte_fallback') or model.get('continuing_subword_prefix')
                or model.get('end_of_word_suffix') or raw.get('normalizer') is not None
                or pre['type'] != 'Sequence' or len(stages) != 2
                or stages[0]['type'] != 'Split' or stages[0]['behavior'] != 'Isolated'
                or stages[0].get('invert', False) or stages[1]['type'] != 'ByteLevel'
                or stages[1]['add_prefix_space'] or stages[1]['use_regex']
                or raw['decoder']['type'] != 'ByteLevel'
                or raw.get('post_processor', {}).get('type') != 'ByteLevel'):
            raise ValueError('Unsupported tokenizer pipeline; expected LFM3 ByteLevel BPE')
        self.pattern = regex.compile(stages[0]['pattern']['Regex'])
        self.vocab = model['vocab']
        self.ranks = {tuple(merge.split(' ') if isinstance(merge, str) else merge): rank
                      for rank, merge in enumerate(model['merges'])}
        self.ignore_merges = model.get('ignore_merges', False)
        self.added = {entry['content']: entry['id'] for entry in raw.get('added_tokens', [])}
        if any(entry.get(flag, False) for entry in raw.get('added_tokens', [])
               for flag in ('single_word', 'lstrip', 'rstrip', 'normalized')):
            raise ValueError('Unsupported added-token normalization or whitespace flags')
        self.special = {entry['id'] for entry in raw.get('added_tokens', []) if entry.get('special')}
        self.ids = {value: key for key, value in {**self.vocab, **self.added}.items()}
        self.added_ids = set(self.added.values())
        alternatives = sorted(self.added, key=len, reverse=True)
        self.added_pattern = re.compile('(' + '|'.join(map(re.escape, alternatives)) + ')')
        # Reversible GPT-2 byte-to-Unicode alphabet used by ByteLevel.
        direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        extra = [byte for byte in range(256) if byte not in direct]
        self.byte_chars = dict(zip(direct + extra,
                                   map(chr, direct + list(range(256, 256 + len(extra))))))
        self.char_bytes = {char: byte for byte, char in self.byte_chars.items()}

    @functools.lru_cache(maxsize=4096)
    def _encode_piece(self, piece):
        word = ''.join(self.byte_chars[byte] for byte in piece.encode('utf-8'))
        if self.ignore_merges and word in self.vocab:
            return (self.vocab[word],)
        parts = list(word)
        while len(parts) > 1:
            choices = ((self.ranks[(a, b)], i) for i, (a, b) in enumerate(zip(parts, parts[1:]))
                       if (a, b) in self.ranks)
            best = min(choices, default=None)
            if best is None:
                break
            _, i = best
            parts[i:i+2] = [parts[i] + parts[i+1]]
        return tuple(self.vocab[part] for part in parts)

    def encode(self, text):
        result = []
        for section in self.added_pattern.split(text):
            if section in self.added:
                result.append(self.added[section])
                continue
            cursor = 0
            for match in self.pattern.finditer(section):
                if match.start() > cursor:
                    result.extend(self._encode_piece(section[cursor:match.start()]))
                result.extend(self._encode_piece(match.group()))
                cursor = match.end()
            if cursor < len(section):
                result.extend(self._encode_piece(section[cursor:]))
        return result

    def decode(self, ids, skip_special_tokens=True):
        parts = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special:
                continue
            word = self.ids[token_id]
            parts.append(word.encode('utf-8') if token_id in self.added_ids
                         else bytes(self.char_bytes[char] for char in word))
        return b''.join(parts).decode('utf-8', errors='replace')


def render_prompt(directory, prompt=PROMPT, enable_thinking=True):
    """Render the downloaded official template for the single user test message."""
    try:
        from jinja2 import nodes
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        raise RuntimeError('Install the chat-template dependency: pip install jinja2') from exc

    class GenerationTag(Extension):
        # The reference tag records assistant spans during training. At inference
        # it simply renders its body; no Transformers extension is needed.
        tags = {'generation'}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(['name:endgeneration'], drop_needle=True)
            return nodes.Scope(body, lineno=lineno)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=[GenerationTag])

    def raise_exception(message):
        raise ValueError(message)

    env.globals['raise_exception'] = raise_exception
    config = read_json(Path(directory) / 'tokenizer_config.json')
    template = env.from_string((Path(directory) / 'chat_template.jinja').read_text(encoding='utf-8'))
    return template.render(messages=[dict(role='user', content=prompt)],
                           bos_token=config['bos_token'], eos_token=config['eos_token'],
                           add_generation_prompt=True, tools=None, enable_thinking=enable_thinking)


# Autoregressive generation with the local PyTorch backbone and LM head.


def sample_next_token(logits, history, temperature, top_k, repetition_penalty, generator=None):
    """HF-compatible penalty -> temperature -> top-k -> multinomial order."""
    scores = logits.float().clone()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError('Nonfinite model logits: local inference failed')
    if repetition_penalty != 1.0:
        seen = torch.tensor(sorted(set(history)), device=scores.device, dtype=torch.long)
        previous = scores[seen]
        scores[seen] = torch.where(previous < 0, previous * repetition_penalty,
                                  previous / repetition_penalty)
    scores = scores / temperature
    if top_k:
        threshold = scores.topk(min(top_k, scores.numel())).values[-1]
        scores.masked_fill_(scores < threshold, float('-inf'))
    probabilities = scores.softmax(-1)
    return torch.multinomial(probabilities, 1, generator=generator).item()


@torch.inference_mode()
def generate(model, prompt_ids, *, eos_token_ids, max_new_tokens=None, progress=False):
    """Sample using the local backbone/head, recomputing each prefix without a cache."""
    max_new_tokens = SAMPLING['max_new_tokens'] if max_new_tokens is None else max_new_tokens
    temperature, top_k, repetition_penalty, seed = (
        SAMPLING[key] for key in ('temperature', 'top_k', 'repetition_penalty', 'seed'))
    if (not prompt_ids or max_new_tokens < 1 or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(repetition_penalty) or repetition_penalty <= 0 or top_k < 0):
        raise ValueError('Need a prompt, positive token limit/temperature/penalty, and top_k >= 0')
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    generated = []
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
    started = time.monotonic()
    model.eval()
    for _ in range(max_new_tokens):
        ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
        if ids.shape[1] > model.model.c['max_position_embeddings']:
            raise ValueError('Generation exceeds the configured context length')
        # Same local policy computation, with the LM head evaluated only at the
        # last position to avoid allocating [prefix_length, 128000] logits.
        hidden = model.model(ids)
        logits = model.lm_head(hidden[:, -1])[0]
        token = sample_next_token(logits, prompt_ids + generated, temperature, top_k,
                                  repetition_penalty, rng)
        generated.append(token)
        if progress and (len(generated) == 1 or len(generated) % 16 == 0 or token in eos):
            print(f'Generated {len(generated)} tokens in {time.monotonic()-started:.1f}s',
                  file=sys.stderr, flush=True)
        if token in eos:
            return generated, 'eos_token'
    return generated, 'max_new_tokens'


def split_response(text, enable_thinking):
    if not enable_thinking:
        return text.strip(), None
    # The opening <think> already belongs to the prompt, not generated token IDs.
    reasoning, separator, content = text.partition('</think>')
    return (content.strip(), reasoning.strip()) if separator else ('', reasoning.strip())


def run_generation(prompt=PROMPT, max_new_tokens=None, offline=False):
    """Verify the real checkpoint, generate one answer, and save the result JSON."""
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    dtype_name = str(dtype).removeprefix('torch.')
    directory, revision = prepare_checkpoint(offline=offline)
    raw_config = read_json(directory / 'config.json')
    validate_model_config(raw_config)
    tokenizer = LocalTokenizer(directory / 'tokenizer.json')
    prompt_text = render_prompt(directory, prompt)
    prompt_ids = tokenizer.encode(prompt_text)
    if tokenizer.decode(prompt_ids, skip_special_tokens=False) != prompt_text:
        raise RuntimeError('Tokenizer roundtrip failed for the exact chat prompt')
    if not prompt_ids or prompt_ids[0] != raw_config['bos_token_id']:
        raise RuntimeError('The official chat prompt must start with the model BOS token')
    print(f'Model: {MODEL_ID}@{revision}\nDevice: {device}, dtype: {dtype_name}, '
          f'prompt: {len(prompt_ids)} tokens\nPrompt: {prompt_text!r}', file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        model, _ = lfm3.initialize(ckpt=directory, device=device, dtype=dtype)
    if sum(p.numel() for p in model.parameters()) != 1728801536:
        raise RuntimeError('Unexpected model parameter count')
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError('Input/output embedding weights must remain tied')
    started = time.monotonic()
    settings = dict(SAMPLING)
    if max_new_tokens is not None:
        settings['max_new_tokens'] = max_new_tokens
    generation_config = read_json(directory / 'generation_config.json')
    generated, stop = generate(model, prompt_ids, eos_token_ids=generation_config['eos_token_id'],
                               max_new_tokens=settings['max_new_tokens'], progress=True)
    raw_text = tokenizer.decode(generated)
    if not raw_text.strip():
        raise RuntimeError('Generation produced no text')
    content, reasoning = split_response(raw_text, enable_thinking=True)
    result = dict(model_id=MODEL_ID, revision=revision, device=device, dtype=dtype_name,
                  messages=[dict(role='user', content=prompt)], sampling=dict(do_sample=True, **settings),
                  prompt_text=prompt_text, prompt_ids=prompt_ids, generated_ids=generated,
                  generated_text=raw_text, assistant_content=content, reasoning_content=reasoning,
                  stop_reason=stop, generated_tokens=len(generated), elapsed_seconds=time.monotonic()-started,
                  checks=dict(config_matches=True, checkpoint_loaded=True, tied_embeddings=True,
                              tokenizer_roundtrip=True, finite_logits=True))
    output = directory / 'generation_result.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(content if content else raw_text, flush=True)
    print(f'Saved generation and checks to {output}; stop={stop}', file=sys.stderr, flush=True)
    if not content:
        print('The token limit was reached during reasoning; assistant_content is empty. '
              'generated_text contains the partial reasoning.', file=sys.stderr)
    return result


# Offline test helpers and cases are defined below. Importing this module never
# downloads a checkpoint or starts a generation run.

def write_safetensors(path, tensors):
    header, parts, offset = {}, [], 0
    names = {dtype: name for name, dtype in lfm3.DTYPES.items()}
    for name, tensor in tensors.items():
        raw = bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())
        header[name] = dict(dtype=names[tensor.dtype], shape=list(tensor.shape),
                            data_offsets=[offset, offset + len(raw)])
        parts.append(raw)
        offset += len(raw)
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data + b''.join(parts))


def reference_rope(q, k, theta, partial, positions):
    """Independent pairwise equation, with full-head spacing and dtype rounding."""
    dim = q.shape[-1]
    pairs = int(partial * dim) // 2

    def apply(x):
        first, second = [], []
        for i in range(dim // 2):
            frequency = theta ** (-2 * i / dim) if i < pairs else 0.0
            angle = positions.float() * frequency
            co = angle.cos().to(x.dtype).float()[:, None]
            si = angle.sin().to(x.dtype).float()[:, None]
            a, b = x[..., i].float(), x[..., i + dim // 2].float()
            first.append(a * co - b * si)
            second.append(b * co + a * si)
        return torch.stack(first + second, dim=-1).to(x.dtype)

    return apply(q), apply(k)


def reference_attention(layer, x, positions):
    """Official-style eager attention equation, independent of SDPA and masks."""
    b, t, _ = x.shape
    q = layer.q_norm(layer.q_proj(x).view(b, t, layer.h, layer.hd)).transpose(1, 2)
    k = layer.k_norm(layer.k_proj(x).view(b, t, layer.kh, layer.hd)).transpose(1, 2)
    v = layer.v_proj(x).view(b, t, layer.kh, layer.hd).transpose(1, 2)
    q, k = reference_rope(q, k, layer.rope['rope_theta'],
                          layer.rope.get('partial_rotary_factor', 1.0), positions)
    k = k.repeat_interleave(layer.h // layer.kh, 1)
    v = v.repeat_interleave(layer.h // layer.kh, 1)
    scores = q @ k.transpose(-1, -2) * layer.hd**-0.5
    mask = torch.ones(t, t, dtype=torch.bool).tril()
    if layer.sliding_window is not None:
        for i in range(t):
            mask[i, :max(0, i - layer.sliding_window + 1)] = False
    scores = scores.masked_fill(~mask, float('-inf'))
    probs = scores.float().softmax(-1).to(q.dtype)
    out = (probs @ v).transpose(1, 2)
    out = out * torch.sigmoid(layer.gate_proj(x)).unsqueeze(-1)
    return layer.o_proj(out.reshape(b, t, -1))


class SeededTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)


class ArchitectureTests(SeededTestCase):
    def test_real_config_and_official_tensor_layout(self):
        c = lfm3.cfg()
        self.assertEqual((c['vocab'], c['dim'], c['ff']), (128000, 2048, 7680))
        self.assertEqual((c['heads'], c['kv_heads'], c['head_dim']), (16, 4, 128))
        self.assertEqual(c['max_position_embeddings'], 262144)
        self.assertEqual(c['layers'].count('conv'), 13)
        self.assertEqual(c['layers'].count('full_attention'), 6)
        self.assertEqual(c['layers'].count('sliding_attention'), 5)
        self.assertEqual((c['pad_token_id'], c['bos_token_id'], c['eos_token_id']),
                         (124893, 124894, 124900))
        with torch.device('meta'):
            model = lfm3.LFMPolicy(c)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1728801536)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        state = model.state_dict()
        self.assertEqual(len(state), 287)
        shapes = {
            'model.layers.0.conv.conv.weight': (2048, 1, 7),
            'model.layers.2.self_attn.q_proj.weight': (2048, 2048),
            'model.layers.2.self_attn.k_proj.weight': (512, 2048),
            'model.layers.2.self_attn.v_proj.weight': (512, 2048),
            'model.layers.2.self_attn.o_proj.weight': (2048, 2048),
            'model.layers.2.self_attn.gate_proj.weight': (16, 2048),
            'model.layers.2.self_attn.q_norm.weight': (128,),
            'model.layers.2.self_attn.k_norm.weight': (128,),
            'model.layers.0.feed_forward.w1.weight': (7680, 2048),
            'model.layers.0.feed_forward.w2.weight': (2048, 7680),
            'model.layers.0.feed_forward.w3.weight': (7680, 2048),
            'model.layers.0.post_operator_norm.weight': (2048,),
            'model.layers.0.post_ffn_norm.weight': (2048,),
        }
        for name, shape in shapes.items():
            self.assertEqual(tuple(state[name].shape), shape, name)
        self.assertFalse(any('rotary' in name or 'bias' in name for name in state))
        self.assertEqual(model.model.embed_tokens.padding_idx, 124893)
        self.assertEqual(set(lfm3.cfg(tiny=True)['layers']),
                         {'conv', 'full_attention', 'sliding_attention'})

    def test_attention_matches_eager_outputs_and_gradients(self):
        c = lfm3.cfg(tiny=True)
        c['sliding_window'] = 3
        positions = torch.tensor([[2, 3, 4, 5, 6, 7]])
        for kind in lfm3.ATTENTION_TYPES:
            layer = lfm3.Attention(c, kind)
            ref = copy.deepcopy(layer)
            x = torch.randn(2, 6, c['dim'], requires_grad=True)
            xr = x.detach().clone().requires_grad_(True)
            actual = layer(x, positions)
            expected = reference_attention(ref, xr, positions)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(x.grad, xr.grad, atol=3e-6, rtol=3e-5)
            for (name, p), (_, pr) in zip(layer.named_parameters(), ref.named_parameters()):
                torch.testing.assert_close(p.grad, pr.grad, atol=3e-6, rtol=3e-5, msg=name)

    def test_short_conv_matches_explicit_causal_equation(self):
        c = lfm3.cfg(tiny=True)
        for bias in (False, True):
            c['conv_bias'] = bias
            conv = lfm3.ShortConv(c)
            x = torch.randn(2, 10, c['dim'], requires_grad=True)
            b, g, z = conv.in_proj(x).chunk(3, -1)
            s = b * z
            outputs = []
            for t in range(x.shape[1]):
                y = torch.zeros_like(s[:, t])
                for j in range(7):
                    if t + j - 6 >= 0:
                        y = y + s[:, t + j - 6] * conv.conv.weight[:, 0, j]
                if bias:
                    y = y + conv.conv.bias
                outputs.append(y)
            expected = conv.out_proj(g * torch.stack(outputs, dim=1))
            actual = conv(x)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertEqual(conv.conv.groups, c['dim'])

    def test_sandwich_norms_surround_each_residual_branch(self):
        c = lfm3.cfg(tiny=True)
        x = torch.randn(2, 4, c['dim'])
        for enabled in (False, True):
            c['use_sandwich_norm'] = enabled
            for kind in ('conv', 'full_attention', 'sliding_attention'):
                layer = lfm3.DecoderLayer(c, kind)
                h = layer.operator_norm(x)
                h = layer.self_attn(h) if layer.is_attn else layer.conv(h)
                if enabled:
                    h = layer.post_operator_norm(h)
                u = x + h
                f = layer.feed_forward(layer.ffn_norm(u))
                if enabled:
                    f = layer.post_ffn_norm(f)
                torch.testing.assert_close(layer(x), u + f)
                self.assertEqual(hasattr(layer, 'post_operator_norm'), enabled)

    def test_invalid_config_inputs_and_real_checkpoint_requirement(self):
        for key, value in (('layers', ['unknown']), ('kv_heads', 3), ('head_dim', 7),
                           ('sliding_window', None), ('pad_token_id', 1000)):
            c = lfm3.cfg(tiny=True)
            c[key] = value
            with self.assertRaises(ValueError, msg=key):
                lfm3.LFMPolicy(c)
        c = lfm3.cfg(tiny=True)
        c['rope_parameters']['full_attention']['rope_type'] = 'dynamic'
        with self.assertRaisesRegex(ValueError, 'default RoPE'):
            lfm3.LFMPolicy(c)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            lfm3.initialize()
        model, _ = lfm3.initialize(tiny=True, device='cpu')
        ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, 'critic=True'):
            model(ids, with_values=True)
        with self.assertRaisesRegex(ValueError, 'position_ids'):
            model(ids, position_ids=torch.tensor([[0., 1., 2.]]))
        with self.assertRaisesRegex(ValueError, 'attention_mask'):
            model(ids, attention_mask=torch.ones(1, 1, 3, 3))



class PositionTests(SeededTestCase):
    def test_partial_rope_coordinates_spacing_and_bfloat16_rounding(self):
        positions = torch.tensor([[0, 23, 262143], [1, 50, 129]])
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(2, 4, 3, 128).to(dtype)
            k = torch.randn(2, 2, 3, 128).to(dtype)
            for theta, partial in ((1e6, 0.5), (1e4, 1.0)):
                actual = lfm3.rotary(q, k, theta, partial, positions)
                # Direct frequency evaluation can round differently at very large
                # positions, so exact pairwise comparison uses moderate positions.
                small = positions.clamp_max(100)
                expected = reference_rope(q, k, theta, partial, small)
                small_actual = lfm3.rotary(q, k, theta, partial, small)
                for a, e in zip(small_actual, expected):
                    torch.testing.assert_close(a, e, atol=0.02 if dtype == torch.bfloat16 else 2e-5,
                                               rtol=0.02 if dtype == torch.bfloat16 else 2e-5)
                if partial == 0.5:
                    for x, rotated in zip((q, k), actual):
                        self.assertTrue(torch.equal(x[..., 32:64], rotated[..., 32:64]))
                        self.assertTrue(torch.equal(x[..., 96:128], rotated[..., 96:128]))
                        self.assertFalse(torch.equal(x[..., 64:96], rotated[..., 64:96]))
                self.assertTrue(torch.equal(actual[0][0, :, 0], q[0, :, 0]))

    def test_sliding_boundary_includes_current_plus_window_minus_one(self):
        c = lfm3.cfg(tiny=True)
        c['layers'] = ['full_attention', 'sliding_attention']
        masks = lfm3._attention_masks(c, 258, 'cpu')
        self.assertIsNone(masks['full_attention'])
        mask = masks['sliding_attention'][0, 0]
        self.assertEqual(mask[255].sum().item(), 256)
        self.assertFalse(mask[256, 0].item())
        self.assertTrue(mask[256, 1].item())
        self.assertTrue(mask[256, 256].item())
        self.assertFalse(mask[256, 257].item())

    def test_padding_with_biased_convolution_and_explicit_positions(self):
        c = lfm3.cfg(tiny=True)
        c['conv_bias'] = True
        c['sliding_window'] = 3
        model = lfm3.LFMPolicy(c)
        ids = torch.tensor([[9, 8, 1, 22, 33, 10], [1, 22, 33, 10, 8, 9]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        logits, _ = model(ids, attention_mask=mask, position_ids=positions)
        expected, _ = model(torch.tensor([[1, 22, 33, 10]]))
        torch.testing.assert_close(logits[0, 2:], expected[0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(logits[1, :4], expected[0], atol=2e-5, rtol=2e-5)
        # Changing masked IDs must not change any kept output.
        changed = ids.clone()
        changed[mask == 0] = 45
        other, _ = model(changed, attention_mask=mask, position_ids=positions)
        torch.testing.assert_close(logits[mask.bool()], other[mask.bool()])
        self.assertTrue(torch.isfinite(logits).all())

    def test_packed_documents_match_separate_forwards_and_gradients(self):
        c = lfm3.cfg(tiny=True)
        c['conv_bias'] = True
        c['sliding_window'] = 2
        model = lfm3.LFMPolicy(c)
        ref = copy.deepcopy(model)
        ids = torch.tensor([[1, 22, 33, 10, 11, 12], [2, 13, 14, 15, 16, 17]])
        positions = torch.tensor([[0, 1, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2]])
        packed, _ = model(ids, position_ids=positions)
        separate = []
        for row, split in ((0, 2), (1, 3)):
            a, _ = ref(ids[row:row+1, :split])
            b, _ = ref(ids[row:row+1, split:])
            separate.append(torch.cat([a, b], dim=1))
        expected = torch.cat(separate, dim=0)
        torch.testing.assert_close(packed, expected, atol=2e-5, rtol=2e-5)
        ones, _ = model(ids, attention_mask=torch.ones_like(ids), position_ids=positions)
        torch.testing.assert_close(packed, ones)
        packed.square().mean().backward()
        expected.square().mean().backward()
        for (name, p), (_, pr) in zip(model.named_parameters(), ref.named_parameters()):
            torch.testing.assert_close(p.grad, pr.grad, atol=3e-5, rtol=5e-5, msg=name)
        with self.assertRaisesRegex(ValueError, 'Packed documents'):
            model(ids[:1], attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
                  position_ids=positions[:1])



class ModelTests(SeededTestCase):
    def test_full_model_causality_gradient_and_value_heads(self):
        model, _ = lfm3.initialize(tiny=True, critic=True, device='cpu')
        ids = torch.tensor([[1, 22, 33, 10, 11, 12, 13, 14, 15]])
        logits, values = model(ids, with_values=True)
        self.assertEqual(tuple(logits.shape), (1, 9, 96))
        self.assertEqual(tuple(values.shape), (1, 9))
        for t in (1, 3, 7):
            lp, vp = model(ids[:, :t], with_values=True)
            torch.testing.assert_close(lp, logits[:, :t], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(vp, values[:, :t], atol=2e-6, rtol=2e-5)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 96), ids[:, 1:].reshape(-1))
        (loss + values.square().mean()).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        critic, _ = lfm3.initialize_critic(tiny=True, device='cpu')
        self.assertFalse(hasattr(critic, 'lm_head'))
        self.assertEqual(tuple(critic(ids).shape), (1, 9))
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertTrue({p.data_ptr() for p in model.parameters()}.isdisjoint(
            {p.data_ptr() for p in critic.parameters()}))

    def test_bfloat16_and_autocast_forward_backward(self):
        ids = torch.tensor([[1, 22, 33, 10]])
        positions = torch.tensor([[0, 1, 0, 1]])
        for dtype in (torch.float32, torch.bfloat16):
            model, _ = lfm3.initialize(tiny=True, critic=True, device='cpu', dtype=dtype)
            self.assertEqual(model.model.embed_tokens.weight.dtype, dtype)
            self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
            for packed in (False, True):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.float32):
                    logits, values = model(ids, with_values=True,
                                           position_ids=positions if packed else None)
                    loss = logits.float().square().mean() + values.float().square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertTrue(torch.isfinite(p.grad).all(), name)



class CheckpointTests(SeededTestCase):
    def test_safetensors_reader_dtypes_and_bad_offsets(self):
        tensors = {name: torch.tensor([[1., 2.]], dtype=dtype)
                   for name, dtype in lfm3.DTYPES.items()}
        tensors['empty'] = torch.empty(0, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.safetensors'
            write_safetensors(path, tensors)
            actual = lfm3.read_safetensors(path)
            for name, tensor in tensors.items():
                self.assertTrue(torch.equal(actual[name], tensor), name)
            data = path.read_bytes()
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, 'offsets'):
                lfm3.read_safetensors(path)

    def test_single_and_sharded_checkpoints_policy_critic_and_missing_names(self):
        original, _ = lfm3.initialize(tiny=True, device='cpu')
        ids = torch.tensor([[1, 22, 33, 10]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for include_head in (False, True):
                state = {n: t for n, t in original.state_dict().items()
                         if include_head or n != 'lm_head.weight'}
                path = root / 'model.safetensors'
                write_safetensors(path, state)
                for sharded in (False, True):
                    if sharded:
                        items = list(state.items())
                        shards = [dict(items[::2]), dict(items[1::2])]
                        weight_map = {}
                        for i, shard in enumerate(shards):
                            name = f'model-{i}.safetensors'
                            write_safetensors(root / name, shard)
                            weight_map.update(dict.fromkeys(shard, name))
                        (root / 'model.safetensors.index.json').write_text(
                            json.dumps(dict(weight_map=weight_map)))
                    ckpt = root if sharded else path
                    actor, _ = lfm3.initialize(tiny=True, ckpt=ckpt, device='cpu')
                    critic, _ = lfm3.initialize_critic(tiny=True, ckpt=ckpt, device='cpu')
                    self.assertIs(actor.lm_head.weight, actor.model.embed_tokens.weight)
                    torch.testing.assert_close(actor(ids)[0], original(ids)[0], atol=0, rtol=0)
                    for name, tensor in original.model.state_dict().items():
                        self.assertTrue(torch.equal(critic.model.state_dict()[name], tensor), name)
                    before = copy.deepcopy(critic.value_head.state_dict())
                    lfm3.load_weights(critic, ckpt)
                    for name, tensor in before.items():
                        self.assertTrue(torch.equal(critic.value_head.state_dict()[name], tensor))
            # A real missing backbone weight or unexpected tensor must fail.
            bad = dict(state)
            bad.pop('model.layers.0.conv.in_proj.weight')
            write_safetensors(path, bad)
            with self.assertRaisesRegex(RuntimeError, 'model.layers.0.conv.in_proj.weight'):
                lfm3.load_weights(actor, path)
            bad = dict(state, unexpected=torch.ones(1))
            write_safetensors(path, bad)
            with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                lfm3.load_weights(actor, path)
            bad = dict(state)
            bad['model.embedding_norm.weight'] = torch.ones(1)
            write_safetensors(path, bad)
            with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
                lfm3.load_weights(actor, path)
            # An index pointing a tensor to the wrong shard cannot silently pass.
            weight_map['model.embedding_norm.weight'] = 'model-0.safetensors'
            if 'model.embedding_norm.weight' in shards[0]:
                weight_map['model.embedding_norm.weight'] = 'model-1.safetensors'
            (root / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=weight_map)))
            with self.assertRaisesRegex(RuntimeError, 'index disagrees'):
                lfm3.load_weights(actor, root)

            # BF16 checkpoint data must copy correctly into FP32 parameters.
            bf16 = {name: tensor.to(torch.bfloat16) for name, tensor in state.items()
                    if name != 'lm_head.weight'}
            write_safetensors(path, bf16)
            actor, _ = lfm3.initialize(tiny=True, ckpt=path, device='cpu')
            for name, tensor in bf16.items():
                self.assertTrue(torch.equal(actor.state_dict()[name], tensor.float()), name)
            self.assertIs(actor.lm_head.weight, actor.model.embed_tokens.weight)

            # An untied custom policy needs an actual head tensor.
            c = lfm3.cfg(tiny=True)
            c['tie_word_embeddings'] = False
            untied = lfm3.LFMPolicy(c)
            with self.assertRaisesRegex(RuntimeError, 'lm_head.weight'):
                lfm3.load_weights(untied, path)

    def test_checkpoint_download_resume_checksum_and_offline_reuse(self):
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 4-9/10'}

        data = b'checkpoint'
        info = dict(size=len(data), lfs=dict(sha256=hashlib.sha256(data).hexdigest()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'model.safetensors'
            (root / 'model.safetensors.part').write_bytes(data[:4])
            with patch.object(sys.modules[__name__], '_request', return_value=Response(data[4:])) as request:
                _download(root, 'revision', path.name, info)
            self.assertEqual(request.call_args.kwargs['Range'], 'bytes=4-')
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse((root / 'model.safetensors.part').exists())
            with patch.object(sys.modules[__name__], '_request') as request:
                _download(root, 'revision', path.name, info)
                request.assert_not_called()
            metadata = dict(model_id=MODEL_ID, revision='revision', files={path.name: info})
            (root / 'download_manifest.json').write_text(json.dumps(metadata))
            with patch.object(sys.modules[__name__], '_request') as request:
                self.assertEqual(prepare_checkpoint(root, offline=True), (root, 'revision'))
                request.assert_not_called()
            path.write_bytes(b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'corrupt'):
                prepare_checkpoint(root, offline=True)



class GenerationTests(SeededTestCase):
    def test_generation_sampling_penalty_temperature_and_top_k(self):
        scores = torch.tensor([2., -2., 1., 0.5])
        # A token seen twice is penalized once; negative logits are multiplied.
        with patch.object(torch, 'multinomial', return_value=torch.tensor([2])) as sample:
            token = sample_next_token(scores, [0, 0, 1], 0.5, 2, 2.0)
        expected = torch.tensor([2., float('-inf'), 2., float('-inf')]).softmax(-1)
        torch.testing.assert_close(sample.call_args.args[0], expected)
        self.assertEqual(token, 2)
        self.assertTrue(torch.equal(scores, torch.tensor([2., -2., 1., 0.5])))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            sample_next_token(torch.tensor([float('nan')]), [0], 1., 0, 1.)

    def test_generation_uses_local_last_logits_stops_on_eos_and_runs_without_grad(self):
        model, _ = lfm3.initialize(tiny=True, device='cpu')
        prompt = [1, 22, 33]
        expected = model(torch.tensor([prompt]))[0][0, -1]
        with patch.object(sys.modules[__name__], 'sample_next_token', return_value=7) as sample:
            generated, stop = generate(model, prompt, eos_token_ids=[7], max_new_tokens=5)
        self.assertEqual(generated, [7])
        self.assertEqual(stop, 'eos_token')
        self.assertEqual(sample.call_count, 1)
        logits = sample.call_args.args[0]
        self.assertFalse(logits.requires_grad)
        torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-5)
        with patch.object(sys.modules[__name__], 'sample_next_token', return_value=10):
            generated, stop = generate(model, prompt, eos_token_ids=[7], max_new_tokens=2)
        self.assertEqual(generated, [10, 10])
        self.assertEqual(stop, 'max_new_tokens')
        with self.assertRaises(ValueError):
            generate(model, prompt, eos_token_ids=[7], max_new_tokens=0)

    def test_generated_reasoning_and_content_are_separated(self):
        self.assertEqual(split_response('reasoning</think> answer', True),
                         ('answer', 'reasoning'))
        self.assertEqual(split_response('unfinished reasoning', True),
                         ('', 'unfinished reasoning'))
        self.assertEqual(split_response(' answer ', False), ('answer', None))


class RunnerTests(SeededTestCase):
    def test_command_modes_run_checks_before_generation(self):
        cases = [([], True, dict(prompt=PROMPT, max_new_tokens=SAMPLING['max_new_tokens'], offline=False)),
                 (['--unit'], True, None),
                 (['--offline', '--prompt', 'Hello', '--max-new-tokens', '8'], True,
                  dict(prompt='Hello', max_new_tokens=8, offline=True)),
                 ([], False, None)]
        for arguments, checks_pass, expected in cases:
            with self.subTest(arguments=arguments, checks_pass=checks_pass):
                with patch.object(sys.modules[__name__], 'run_unit_tests', return_value=checks_pass) as checks:
                    with patch.object(sys.modules[__name__], 'run_generation') as generation:
                        self.assertEqual(main(arguments), 0 if checks_pass else 1)
                checks.assert_called_once_with()
                if expected is None:
                    generation.assert_not_called()
                else:
                    generation.assert_called_once_with(**expected)

    def test_invalid_token_limit_fails_before_checks_or_download(self):
        with patch.object(sys.modules[__name__], 'run_unit_tests') as checks:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(['--max-new-tokens', '0'])
        self.assertEqual(error.exception.code, 2)
        checks.assert_not_called()


# Entry point: quick checks first, then optional real-checkpoint generation.


def run_unit_tests():
    """Run every offline case without downloading files; return whether all pass."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        with contextlib.redirect_stdout(sys.stderr):
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        return result.wasSuccessful()
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Run LFM3 checks, then generate with the local PyTorch model.',
        epilog='Advanced defaults live in SAMPLING and DEFAULT_DIRECTORY at the top of this file.')
    parser.add_argument('--unit', action='store_true', help='Run only the fast offline tests; requires only torch')
    parser.add_argument('--offline', action='store_true', help='Generate using verified local files without network access')
    parser.add_argument('--prompt', default=PROMPT, help=f'User message (default: {PROMPT})')
    parser.add_argument('--max-new-tokens', type=int, default=SAMPLING['max_new_tokens'],
                        help=f'Generation token limit (default: {SAMPLING["max_new_tokens"]})')
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if not run_unit_tests():
        return 1
    if not args.unit:
        run_generation(prompt=args.prompt, max_new_tokens=args.max_new_tokens, offline=args.offline)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'LFM3 test failed: {error}', file=sys.stderr)
        raise SystemExit(1)


