"""Loading Pythia checkpoints and reading the residual stream.

The activation site everywhere is the output of transformer block `layer` (after its residual update, before any final
layer norm), captured with forward hooks. This is also why we do not use `output_hidden_states`: its last entry is
taken after the final layer norm.
"""
from contextlib import contextmanager

import numpy as np
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


def n_layers(model):
    return len(model.base_model.layers)


@contextmanager
def block_outputs(model, layers):
    """Within the context, `captured[layer]` holds the output [batch, seq, d] of each block in `layers` from the last
    forward pass."""
    captured, handles = {}, []
    for layer in layers:
        def hook(_module, _inputs, output, layer=layer):
            captured[layer] = output[0] if isinstance(output, tuple) else output
        handles.append(model.base_model.layers[layer].register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def word_representations(model, tokenizer, data, layers, batch_size=32, max_length=256):
    """Activation of each block in `layers` at the last sub-word token of every labelled word, as {layer: float16
    [n_words, d]} on the CPU. Sentences are fed whole (unlabelled words included), `batch_size` at a time, in corpus
    order."""
    device = next(model.parameters()).device
    words_of = [[] for _ in data.sentences]
    for i, s in enumerate(data.sentence_index):
        words_of[s].append(i)
    d = model.config.hidden_size
    reps = {layer: torch.empty(len(data.forms), d, dtype=torch.float16) for layer in layers}
    filled = torch.zeros(len(data.forms), dtype=torch.bool)
    for start in range(0, len(data.sentences), batch_size):
        batch = data.sentences[start:start + batch_size]
        enc = tokenizer(batch, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True,
                        max_length=max_length)
        with block_outputs(model, layers) as captured:
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
        for layer in layers:
            reps[layer][rows] = captured[layer][batch_pos, token_pos].to(torch.float16).cpu()
        filled[rows] = True
    if not filled.all():
        raise RuntimeError('Some words got no representation')
    return reps


@torch.no_grad()
def token_representations(model, lines, line_of, position, layers, pad_id, batch_size=32):
    """Activation of each block in `layers` at given token positions, as {layer: float16 [n, d]} on the CPU. `lines` are
    token-id lists; sample i is token `position[i]` of line `line_of[i]`. Lines are fed `batch_size` at a time in order
    (left padding), from line 0 to the last line with a sample."""
    device = next(model.parameters()).device
    order = np.argsort(line_of, kind='stable')
    d = model.config.hidden_size
    reps = {layer: torch.empty(len(line_of), d, dtype=torch.float16) for layer in layers}
    last_line = int(np.max(line_of))
    start_of = np.searchsorted(line_of[order], np.arange(last_line + 2))
    for start in range(0, last_line + 1, batch_size):
        batch = lines[start:start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, ids in enumerate(batch):
            input_ids[b, width - len(ids):] = torch.tensor(ids)
            mask[b, width - len(ids):] = 1
        with block_outputs(model, layers) as captured:
            model.base_model(input_ids.to(device), attention_mask=mask.to(device))
        rows = order[start_of[start]:start_of[min(start + batch_size, last_line + 1)]]
        if not len(rows):
            continue
        b = torch.from_numpy(line_of[rows] - start)
        t = torch.from_numpy(position[rows]) + torch.tensor([width - len(lines[i]) for i in line_of[rows]])
        for layer in layers:
            reps[layer][torch.from_numpy(rows)] = captured[layer][b, t].to(torch.float16).cpu()
    return reps
