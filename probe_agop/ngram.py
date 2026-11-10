"""The five n-gram regression tasks: data, targets, training/test split and control targets.

Statistics are counted on WikiText-103 with the Pythia tokenizer (no special tokens, lines not truncated):
  unigram counts         over every line with at least one token and no URL-like token ("loose" lines);
  bigram/trigram counts  over the lines that pass the sentence filter of the paper ("strict" lines), within lines.
Probe tokens come from 20,000 strict lines that are not in the AGOP corpus (random.Random(42)), each truncated to 256
tokens. Each position t >= n - 1 of a line is one sample (n = the order of the task); the first N positions of the first
lines are used. Targets (natural log, unsmoothed; every probe n-gram occurs in the counts):
  unigram_lp    log p(w_t)
  bigram_lp     log p(w_t | w_{t-1})
  bigram_pmi    log p(w_t | w_{t-1}) - log p(w_t)                    (p(w_t) over bigram positions)
  trigram_lp    log p(w_t | w_{t-2} w_{t-1})
  trigram_pmi   log p(w_t | w_{t-2} w_{t-1}) - log p(w_t | w_{t-1})
Training/test split by n-gram type (the token for unigram_lp, the bigram or trigram otherwise), so the test set holds only
types unseen in training: keys occurring at least twice are shuffled with torch.Generator().manual_seed(42) and 80% go to
training; for bigram and trigram types, test types whose context or last token never occurs in training are dropped.
Control targets (label seeds 42-44, one np.random.default_rng(seed) for the training set, then the test set): every
n-gram type of a set gets the target of one random sample of the same set (all samples of the type share it).
Targets are centered by subtracting their training-set mean.
"""
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import data

TASKS = {   # n-gram order, target, probe lines, probe samples
    'unigram_lp': (1, 'lp', 5000, 50000),
    'bigram_lp': (2, 'lp', 5000, 50000),
    'bigram_pmi': (2, 'pmi', 5000, 50000),
    'trigram_lp': (3, 'lp', 20000, 500000),
    'trigram_pmi': (3, 'pmi', 20000, 500000),
}
PROBE_LINES, PROBE_SEED, MAX_LENGTH = 20000, 42, 256
SPLIT_SEED, TRAIN_FRACTION, MIN_KEY_COUNT = 42, 0.8, 2
STATS_VERSION = 'wikitext-103-v1/pythia-tokenizer/v1'


def wikitext_pool(root):
    """All non-empty, URL-free lines of the four WikiText-103 parquet files (whitespace-normalized, sorted file-name order),
    and for each whether it passes the sentence filter."""
    import pyarrow.parquet as pq
    lines, strict = [], []
    for name in sorted(n for n in data.FILES if n.startswith('wikitext-103-v1/')):
        for cell in pq.read_table(data.data_file(root, name), columns=['text']).column('text').to_pylist():
            if cell is None:
                continue
            for line in cell.splitlines():
                tokens = line.split()
                if not tokens or any(data.URL_RE.search(t) for t in tokens):
                    continue
                lines.append(' '.join(tokens))
                strict.append(data.keep_sentence(tokens))
    return lines, np.array(strict, dtype=bool)


def probe_corpus(lines, strict):
    """The 20,000 probe lines: strict lines outside the AGOP corpus, drawn with random.Random(42), in corpus order."""
    pool = [line for line, s in zip(lines, strict) if s]
    agop = set(random.Random(data.AGOP_CORPUS_SEED).sample(range(len(pool)), data.AGOP_CORPUS_SIZE))
    rest = [i for i in range(len(pool)) if i not in agop]
    return [pool[i] for i in sorted(random.Random(PROBE_SEED).sample(rest, PROBE_LINES))]


def _tokenize(tokenizer, lines, chunk=20000):
    ids, lengths = [], []
    for i in range(0, len(lines), chunk):
        enc = tokenizer(lines[i:i + chunk], add_special_tokens=False)['input_ids']
        lengths.extend(len(e) for e in enc)
        ids.append(np.fromiter((t for e in enc for t in e), dtype=np.int64))
    return np.concatenate(ids), np.array(lengths, dtype=np.int64)


def count_statistics(tokenizer, lines, strict):
    """Unigram counts (loose lines); previous-token, current-token, bigram, trigram-prefix and trigram counts (strict
    lines, within lines). n-gram counts are kept as sorted codes with counts."""
    V = len(tokenizer)
    ids, lengths = _tokenize(tokenizer, lines)
    S = dict(V=V, unigram=np.bincount(ids, minlength=V), n_tokens=ids.size)
    keep = np.repeat(strict, lengths)
    ids, lengths = ids[keep], lengths[strict]
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    pos = np.arange(ids.size) - starts
    j = np.nonzero(pos >= 1)[0]
    S.update(previous=np.bincount(ids[j - 1], minlength=V), current=np.bincount(ids[j], minlength=V), n_bigrams=j.size)
    S['bigram_keys'], S['bigram_counts'] = np.unique(ids[j - 1] * V + ids[j], return_counts=True)
    j = np.nonzero(pos >= 2)[0]
    S['prefix_keys'], S['prefix_counts'] = np.unique(ids[j - 2] * V + ids[j - 1], return_counts=True)
    S['trigram_keys'], S['trigram_counts'] = np.unique((ids[j - 2] * V + ids[j - 1]) * V + ids[j], return_counts=True)
    return S


def _lookup(keys, counts, query):
    i = np.minimum(np.searchsorted(keys, query), keys.size - 1)
    return np.where(keys[i] == query, counts[i], 0)


@lru_cache(maxsize=None)
def statistics(root, tokenizer):
    lines, strict = wikitext_pool(root)
    return count_statistics(tokenizer, lines, strict)


@lru_cache(maxsize=None)
def probe_lines(root, tokenizer):
    """Token ids of the 20,000 probe lines, each truncated to its first 256 tokens."""
    path = Path(root)/'ngram'/'probe_lines.npz'
    if path.exists():
        z = np.load(path)
        if str(z['version']) != STATS_VERSION or int(z['vocab']) != len(tokenizer):
            raise RuntimeError(f'{path} was made with other settings; delete it to recompute')
        return [ids.tolist() for ids in np.split(z['ids'], np.cumsum(z['lengths'])[:-1])]
    corpus = probe_corpus(*wikitext_pool(root))
    lines = [tokenizer(line, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)['input_ids'] for line in corpus]
    data.write_atomic(path, lambda f: np.savez(f, version=STATS_VERSION, vocab=len(tokenizer),
                                               lengths=np.array([len(x) for x in lines]),
                                               ids=np.concatenate([np.asarray(x, dtype=np.int64) for x in lines])))
    return lines


def probe_stream(lines, order, n_lines, n_positions):
    """The probe samples: for each, its line, the position t of its last token, and the tokens w_{t-n+1} .. w_t. Every
    position t >= n - 1 of the first lines, until n_positions samples."""
    line_of, pos_of, cols, total = [], [], [[] for _ in range(order)], 0
    for li, ids in enumerate(lines[:n_lines]):
        if len(ids) < order:
            continue
        n = len(ids) - (order - 1)
        line_of.append(np.full(n, li, dtype=np.int64))
        pos_of.append(np.arange(order - 1, len(ids), dtype=np.int64))
        for k in range(order):
            cols[k].append(np.asarray(ids[k:k + n], dtype=np.int64))
        total += n
        if total >= n_positions:
            break
    if total < n_positions:
        raise RuntimeError(f'Only {total} probe positions')
    cut = lambda parts: np.concatenate(parts)[:n_positions]
    return cut(line_of), cut(pos_of), [cut(c) for c in cols]


def targets(task, cols, S):
    """Raw targets (float32) with the float arithmetic of the paper runs (float64 logs, rounded to float32)."""
    order, kind = TASKS[task][:2]
    V = S['V']
    if order == 1:
        with np.errstate(divide='ignore'):
            log_p = np.log(S['unigram'].astype(np.float64) / float(S['n_tokens'])).astype(np.float32)
        return log_p[cols[0]]
    if order == 2:
        prev, cur = cols
        joint = _lookup(S['bigram_keys'], S['bigram_counts'], prev * V + cur)
        y = np.log(joint / S['previous'][prev]).astype(np.float32)
        if kind == 'pmi':
            y = (y.astype(np.float64) - np.log(S['current'][cur] / S['n_bigrams'])).astype(np.float32)
        return y
    p2, p1, cur = cols
    tri = _lookup(S['trigram_keys'], S['trigram_counts'], (p2 * V + p1) * V + cur)
    pre = _lookup(S['prefix_keys'], S['prefix_counts'], p2 * V + p1)
    y = np.log(tri / pre)
    if kind == 'pmi':
        y = y - np.log(_lookup(S['bigram_keys'], S['bigram_counts'], p1 * V + cur) / S['previous'][p1])
    return y.astype(np.float32)


def split(order, cols):
    """Training and test sample indices (ascending), split by n-gram key."""
    g = torch.Generator().manual_seed(SPLIT_SEED)
    cur = torch.from_numpy(cols[-1])
    if order == 1:
        counts = torch.bincount(cur, minlength=int(cur.max()) + 1)
        eligible = torch.nonzero(counts >= MIN_KEY_COUNT).view(-1)
        perm = torch.randperm(eligible.numel(), generator=g)
        n_train = min(max(int(TRAIN_FRACTION * eligible.numel()), 1), eligible.numel() - 1)
        train_keys, eval_keys, keys = eligible[perm[:n_train]], eligible[perm[n_train:]], cur
    else:
        if order == 2:
            keys = (torch.from_numpy(cols[0]) << 32) | cur
        else:
            keys = torch.from_numpy(np.unique(np.stack(cols, 1), axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64))
        unique, counts = torch.unique(keys, return_counts=True)
        eligible = unique[counts >= MIN_KEY_COUNT]
        perm = eligible[torch.randperm(eligible.numel(), generator=g)]
        n_train = min(max(1, int(round(TRAIN_FRACTION * perm.numel()))), perm.numel() - 1)
        train_keys, eval_keys = perm[:n_train], perm[n_train:]
        # Drop test keys whose context or last token never occurs in training.
        in_train = torch.isin(keys, train_keys).numpy()
        seen_context = set(zip(*(c[in_train].tolist() for c in cols[:-1])))
        seen_last = set(cols[-1][in_train].tolist())
        first = {}
        for i, k in enumerate(keys.tolist()):
            first.setdefault(k, i)
        kept = [k for k in eval_keys.tolist()
                if tuple(int(c[first[k]]) for c in cols[:-1]) in seen_context and int(cols[-1][first[k]]) in seen_last]
        if kept:
            eval_keys = torch.tensor(kept, dtype=torch.long)
    train = torch.where(torch.isin(keys, train_keys))[0].numpy().astype(np.int64)
    test = torch.where(torch.isin(keys, eval_keys))[0].numpy().astype(np.int64)
    return train, test


def key_index(cols):
    keys = np.stack(cols, 1)
    return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)


def pool_control(y, key, train, test, seed):
    """Control targets (float32): within the training set and within the test set, every key takes the target of one random
    sample of the same set."""
    out = np.full(y.size, np.nan)
    rng = np.random.default_rng(seed)
    for rows in (train, test):
        values = y[rows].astype(np.float64)
        _, local = np.unique(key[rows], return_inverse=True)
        local = local.reshape(-1)
        draw = rng.integers(0, rows.size, size=int(local.max()) + 1)
        out[rows] = values[draw][local]
    return out.astype(np.float32)


def centre_targets(y, train):
    """Targets minus their training-set mean (float64 mean), float32."""
    y = torch.from_numpy(y).double()
    return (y - y[torch.from_numpy(train)].mean()).float()
