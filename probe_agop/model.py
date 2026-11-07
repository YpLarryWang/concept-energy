"""Loading Pythia checkpoints and reading the residual stream.

The activation site everywhere is the output of transformer block `layer` (after its residual update, before any final
layer norm), captured with a forward hook. This is also why we do not use `output_hidden_states`: its last entry is
taken after the final layer norm.
"""
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name):
    # Left padding and pad = eos, as in the paper runs (nnsight's default). Pythia uses rotary position embeddings, which
    # depend only on relative positions, so the padding side does not change the activations of real tokens.
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name, step, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(name, revision=f'step{step}', dtype=dtype).to(device).eval()
    if next(model.parameters()).dtype != dtype:
        raise RuntimeError(f'Model did not load in {dtype}; use the transformers version in requirements.txt')
    return model


@contextmanager
def block_output(model, layer):
    """Within the context, `captured['h']` holds the output [batch, seq, d] of block `layer` from the last forward."""
    captured = {}

    def hook(_module, _inputs, output):
        captured['h'] = output[0] if isinstance(output, tuple) else output

    handle = model.base_model.layers[layer].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@torch.no_grad()
def word_representations(model, tokenizer, data, layer, batch_size=32, max_length=256):
    """Block-`layer` activation at the last sub-word token of every labelled word, float32 [n_words, d].
    Sentences are fed whole (punctuation included), `batch_size` at a time, in corpus order."""
    device = next(model.parameters()).device
    words_of = [[] for _ in data.sentences]
    for i, s in enumerate(data.sentence_index):
        words_of[s].append(i)
    reps = torch.empty(len(data.words), model.config.hidden_size, dtype=torch.float32)
    filled = torch.zeros(len(data.words), dtype=torch.bool)
    for start in range(0, len(data.sentences), batch_size):
        batch = data.sentences[start:start + batch_size]
        enc = tokenizer(batch, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True,
                        max_length=max_length)
        with block_output(model, layer) as captured:
            model.base_model(enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device))
        rows, batch_pos, token_pos = [], [], []
        for b in range(len(batch)):
            last_token = {}
            for t, word_pos in enumerate(enc.word_ids(b)):
                if word_pos is not None:
                    last_token[word_pos] = t
            for i in words_of[start + b]:
                rows.append(i)
                batch_pos.append(b)
                token_pos.append(last_token[data.position[i]])   # KeyError if the word was truncated away
        reps[rows] = captured['h'][batch_pos, token_pos].float().cpu()
        filled[rows] = True
    if not filled.all():
        raise RuntimeError('Some words got no representation')
    return reps
