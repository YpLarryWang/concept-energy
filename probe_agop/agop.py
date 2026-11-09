"""Average gradient outer product (AGOP) of the language-modelling loss at residual-stream sites.

    G_layer = mean over sites s of  g_s g_s^T,   g_s = d loss_s / d h_layer[position of s]

A site is a (sequence, position) pair; loss_s is the next-token cross-entropy at that position, and h_layer is the output
of block `layer` at the same position. As in the paper: 1,000 WikiText-103 lines truncated to 128 tokens, 8 positions
per line sampled once with np.random.RandomState(42) (fixed across checkpoints and layers), every site weighted equally.
The AGOP depends only on the model, the checkpoint and the layer, so all tasks share it.
Pass a float32 model: in float16 the backward pass loses several percent of G at late checkpoints.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import block_outputs

TOKENS_PER_SEQUENCE, MAX_LENGTH, POSITION_SEED = 8, 128, 42


def sampled_positions(texts, tokenizer, k=TOKENS_PER_SEQUENCE, max_length=MAX_LENGTH, seed=POSITION_SEED):
    """For each text, k distinct positions p (sorted) whose next token p + 1 exists; [] for texts shorter than 2 tokens."""
    rng = np.random.RandomState(seed)
    positions = []
    for text in texts:
        n = len(tokenizer(text, truncation=True, max_length=max_length)['input_ids'])
        if n < 2:
            positions.append([])
            continue
        positions.append(sorted(rng.choice(n - 1, size=min(k, n - 1), replace=False).tolist()))
    return positions


def compute_agops(model, tokenizer, layers, texts, batch_size=8, max_length=MAX_LENGTH):
    """AGOP [d, d] (float64 numpy) of `model` at the output of each block in `layers`, as {layer: G}."""
    device = next(model.parameters()).device
    positions = sampled_positions(texts, tokenizer, max_length=max_length)
    ids = [tokenizer(text, truncation=True, max_length=max_length)['input_ids'] for text in texts]
    order = sorted(range(len(texts)), key=lambda i: len(ids[i]))      # similar lengths share a batch (less padding)
    d = model.config.hidden_size
    G = {layer: torch.zeros(d, d, dtype=torch.float64, device=device) for layer in layers}
    n_sites = 0
    for start in range(0, len(order), batch_size):
        batch = [i for i in order[start:start + batch_size] if positions[i]]
        if not batch:
            continue
        width = max(len(ids[i]) for i in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, i in enumerate(batch):                                    # right padding: positions need no shift
            input_ids[b, :len(ids[i])] = torch.tensor(ids[i])
            mask[b, :len(ids[i])] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)
        with block_outputs(model, layers) as captured:
            logits = model(input_ids, attention_mask=mask).logits
        hs = [captured[layer] for layer in layers]
        # The j-th sampled position of every sequence in one backward pass: sequences do not interact, so the gradient
        # of the summed loss at sequence b equals the gradient of b's own loss.
        for j in range(max(len(positions[i]) for i in batch)):
            rows = [b for b, i in enumerate(batch) if j < len(positions[i])]
            pos = [positions[batch[b]][j] for b in rows]
            loss = F.cross_entropy(logits[rows, pos].float(), input_ids[rows, [p + 1 for p in pos]], reduction='sum')
            grads = torch.autograd.grad(loss, hs, retain_graph=True)
            for layer, grad in zip(layers, grads):
                g = grad[rows, pos].double()
                G[layer] += g.T @ g
            n_sites += len(rows)
        del logits, hs, captured
    out = {}
    for layer in layers:
        g = G[layer].cpu() / n_sites
        out[layer] = (0.5 * (g + g.T)).numpy()
    return out
