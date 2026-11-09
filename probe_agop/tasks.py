"""The 12 probing tasks of the paper.

Classification (word level; probe = softmax regression, score = test accuracy, k = n_classes - 1):
  pos               UPOS tag of every non-punctuation word (UD English EWT)
  dep_relation      relation to the head, subtype stripped (UD English EWT)
  dep_length        direction and distance of the head: ROOT, LEFT/RIGHT x {1, 2, 3-5, 6+} (UD English EWT)
  dep_depth         number of arcs up to the root, 0..10 and 11+ (UD English EWT)
  ner               entity class of entity tokens: PER, ORG, LOC (UNER English EWT)
  noun_supersense   supersense of single-word noun expressions (STREUSLE)
  verb_supersense   supersense of single-word verb expressions (STREUSLE)
A class is kept when it has at least 100 training and 10 test words (words of dropped classes are removed); the
sentences stay whole. The control of every classification task is the type-matched control (controls.py).

Regression (token level; probe = linear regression, score = held-out R^2, k = 1; see ngram.py):
  unigram_lp        log p(w_t)
  bigram_lp         log p(w_t | w_{t-1})
  bigram_pmi        log p(w_t | w_{t-1}) - log p(w_t)
  trigram_lp        log p(w_t | w_{t-2} w_{t-1})
  trigram_pmi       log p(w_t | w_{t-2} w_{t-1}) - log p(w_t | w_{t-1})
read at the last token w_t of the n-gram, with its WikiText-103 line as context. The control is the type-matched pool
control (ngram.pool_control).
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import controls, data, energy, ngram, probe
from . import model as lm

CONTROL_SEEDS = (42, 43, 44)
MIN_TRAIN, MIN_TEST = 100, 10


@lru_cache(maxsize=None)
def _family(root, family):
    """Train and test Words of a dataset; several tasks share one."""
    read = dict(ud_ewt=data.ud_ewt_words, uner_ewt=data.uner_words,
                streusle_noun=lambda r, s: data.streusle_words(r, s, 'n.'),
                streusle_verb=lambda r, s: data.streusle_words(r, s, 'v.'))[family]
    return read(root, 'train'), read(root, 'test')


@dataclass
class WordTaskData:
    """A classification task: its rows of a shared word dataset, labels, and control labels."""
    key: str                 # tasks with the same key share representations
    words: dict              # split -> data.Words of the shared dataset
    rows: dict               # split -> indices of this task's words in words[split]
    y_train: torch.Tensor
    y_test: torch.Tensor
    controls: dict           # control label seed -> (train labels, test labels)
    classes: list
    k: int

    def representations(self, model, tokenizer, layers):
        return {s: lm.word_representations(model, tokenizer, self.words[s], layers) for s in ('train', 'test')}

    def select(self, reps, layer):
        return reps['train'][layer][self.rows['train']].float(), reps['test'][layer][self.rows['test']].float()


@dataclass
class TokenTaskData:
    """An n-gram task: the token positions it reads (train and eval pool), centred targets, and control targets."""
    key: str
    lines: list              # token ids of the probe lines
    line_of: np.ndarray      # per sample (train pool, then eval pool): its line and position
    position: np.ndarray
    n_train: int
    y_train: torch.Tensor
    y_test: torch.Tensor
    controls: dict           # control label seed -> (train targets, eval targets), each centred by its train-pool mean
    k: int = 1

    def representations(self, model, tokenizer, layers):
        return lm.token_representations(model, self.lines, self.line_of, self.position, layers, tokenizer.pad_token_id)

    def select(self, reps, layer):
        X = reps[layer]
        return X[:self.n_train].float(), X[self.n_train:].float()


class Classification:
    unit = 'word: block output at its last sub-word token, whole sentence as context'
    score_name = 'Test accuracy'

    def __init__(self, name, title, family, tag, classes, label=lambda v: v, drop=(), sizes=None):
        self.name, self.title, self.family, self.tag = name, title, family, tag
        self.classes, self.label, self.drop, self.sizes = classes, label, drop, sizes

    def data_fingerprint(self):
        return dict(family=self.family, tag=self.tag, sizes=self.sizes,
                    files={n: h for n, h in data.FILES.items() if n.startswith(self.family.split('_')[0])})

    def recipe(self):
        return dict(kind='classification', lr=probe.LR, weight_decay=probe.WEIGHT_DECAY, batch_size=probe.BATCH_SIZE,
                    epochs=probe.EPOCHS, inputs='centred with the training mean', subspace='top n_classes - 1 right '
                    'singular vectors of the class-centred weights', min_train=MIN_TRAIN, min_test=MIN_TEST)

    def load(self, root, tokenizer=None):
        words = dict(zip(('train', 'test'), _family(str(root), self.family)))
        labels = {s: [self.label(v) for v in words[s].tags[self.tag]] for s in words}
        candidates = self.classes(labels['train']) if callable(self.classes) else self.classes
        count = {s: {c: 0 for c in candidates} for s in words}
        for s in words:
            for v in labels[s]:
                if v in count[s]:
                    count[s][v] += 1
        kept = [c for c in candidates if count['train'][c] >= MIN_TRAIN and count['test'][c] >= MIN_TEST
                and c not in self.drop]
        index = {c: i for i, c in enumerate(kept)}
        rows = {s: np.array([i for i, v in enumerate(labels[s]) if v in index], dtype=np.int64) for s in words}
        y = {s: np.array([index[labels[s][i]] for i in rows[s]], dtype=np.int64) for s in words}
        got = (len(kept), len(y['train']), len(y['test']))
        if self.sizes is not None and got != self.sizes:
            raise ValueError(f'{self.name}: (classes, train, test) = {got}, paper {self.sizes}')
        forms = {s: [words[s].forms[i] for i in rows[s]] for s in words}
        ctrl = {}
        for seed in CONTROL_SEEDS:
            tr, te = controls.type_matched_labels(forms['train'], y['train'], forms['test'], len(kept), seed)
            ctrl[seed] = (torch.from_numpy(tr), torch.from_numpy(te))
        return WordTaskData(key=self.family, words=words, rows=rows, y_train=torch.from_numpy(y['train']),
                        y_test=torch.from_numpy(y['test']), controls=ctrl, classes=kept, k=len(kept) - 1)

    def check_sizes(self, root):
        self.load(root)

    def fit(self, td, X_train, y_train, X_test, y_test, seed, device):
        """Trains one probe; returns its subspace U [d, k] and its test accuracy."""
        p = probe.train_probe(X_train, y_train, len(td.classes), seed, device)
        U = energy.probe_subspace(p.weight.detach().double().cpu().numpy(), td.k)
        return U, probe.accuracy(p, X_test, y_test)


@lru_cache(maxsize=None)
def _ngram_family(root, order, tokenizer):
    """Probe samples of one n-gram order, their raw targets for every task of that order, and the split."""
    path = Path(root)/'ngram'/f'order{order}.npz'
    meta = dict(version=ngram.STATS_VERSION, files={n: h for n, h in data.FILES.items() if n.startswith('wikitext')},
                vocab=len(tokenizer), tasks={n: t for n, t in ngram.TASKS.items() if t[0] == order})
    lines = ngram.probe_lines(root, tokenizer)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if json.loads(str(z['meta'])) != json.loads(json.dumps(meta)):    # compare in JSON form (tuples become lists)
            raise RuntimeError(f'{path} was made with other settings; delete it to recompute')
        out = {k: z[k] for k in z.files if k != 'meta'}
    else:
        print(f'Counting n-grams on WikiText-103 for the order-{order} tasks (once; cached in {path})', flush=True)
        S = ngram.statistics(root, tokenizer)
        n_lines, n_positions = next(t[2:] for t in ngram.TASKS.values() if t[0] == order)
        line_of, position, cols = ngram.probe_stream(lines, order, n_lines, n_positions)
        out = dict(line_of=line_of, position=position, **{f'col{i}': c for i, c in enumerate(cols)})
        out.update({f'y_{n}': ngram.targets(n, cols, S) for n in meta['tasks']})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, meta=json.dumps(meta), **out)
    cols = [out[f'col{i}'] for i in range(order)]
    train, test = ngram.split(order, cols)
    return lines, out, cols, train, test


class Regression:
    unit = 'token: block output at the last token of the n-gram, its WikiText-103 line (first 256 tokens) as context'
    score_name = 'Held-out R²'

    def __init__(self, name, title):
        self.name, self.title = name, title
        self.order = ngram.TASKS[name][0]

    def data_fingerprint(self):
        return dict(task=ngram.TASKS[self.name], statistics=ngram.STATS_VERSION,
                    files={n: h for n, h in data.FILES.items() if n.startswith('wikitext')})

    def recipe(self):
        return dict(kind='regression', loss='mean squared error', lr=probe.LR, weight_decay=probe.WEIGHT_DECAY,
                    batch_size=probe.BATCH_SIZE, epochs=probe.EPOCHS, inputs='centred with the train-pool mean',
                    targets='centred with the train-pool mean', subspace='the normalised weight vector (k = 1)')

    def load(self, root, tokenizer):
        lines, out, cols, train, test = _ngram_family(str(root), self.order, tokenizer)
        y = out[f'y_{self.name}']
        key = ngram.key_index(cols)
        rows = np.concatenate([train, test])
        centred = lambda v: ngram.centre_targets(v, train)[torch.from_numpy(rows)]
        yc = centred(y)
        ctrl = {}
        for seed in CONTROL_SEEDS:
            c = centred(ngram.pool_control(y, key, train, test, seed))
            ctrl[seed] = (c[:train.size], c[train.size:])
        return TokenTaskData(key=f'ngram_order{self.order}', lines=lines, line_of=out['line_of'][rows],
                             position=out['position'][rows], n_train=train.size, y_train=yc[:train.size],
                             y_test=yc[train.size:], controls=ctrl)

    def check_sizes(self, root):
        pass

    def fit(self, td, X_train, y_train, X_test, y_test, seed, device):
        """Trains one probe; returns its direction U [d, 1] and its held-out R^2."""
        p = probe.train_regression(X_train, y_train, seed, device)
        return energy.probe_direction(p.weight.detach().double().cpu().numpy()), probe.r2(p, X_test, y_test)


DEPTH = [str(i) for i in range(11)] + ['11+']
ARC = ['ROOT', 'LEFT_1', 'LEFT_2', 'LEFT_3_5', 'LEFT_6P', 'RIGHT_1', 'RIGHT_2', 'RIGHT_3_5', 'RIGHT_6P']
ENTITY = ['O', 'PER', 'ORG', 'LOC', 'MISC']

TASKS = {}
for _task in [
        Classification('pos', 'POS', 'ud_ewt', 'upos', classes=lambda train: sorted(set(train)), sizes=(16, 176838, 20706)),
        Classification('dep_relation', 'Dependency relation', 'ud_ewt', 'relation', classes=lambda train: sorted(set(train)),
                       sizes=(30, 176696, 20691)),
        Classification('dep_length', 'Dependency length', 'ud_ewt', 'arc', classes=ARC, sizes=(9, 176838, 20706)),
        Classification('dep_depth', 'Dependency depth', 'ud_ewt', 'depth', classes=DEPTH,
                       label=lambda d: str(d) if d < 11 else '11+', sizes=(12, 176838, 20706)),
        Classification('ner', 'NER', 'uner_ewt', 'entity', classes=ENTITY, drop=('O',), sizes=(3, 9233, 1365)),
        Classification('noun_supersense', 'Noun supersense', 'streusle_noun', 'supersense',
                       classes=data.SUPERSENSES['n.'], sizes=(12, 5685, 743)),
        Classification('verb_supersense', 'Verb supersense', 'streusle_verb', 'supersense',
                       classes=data.SUPERSENSES['v.'], sizes=(9, 5344, 592)),
        Regression('unigram_lp', 'Unigram LP'),
        Regression('bigram_lp', 'Bigram LP'),
        Regression('bigram_pmi', 'Bigram PMI'),
        Regression('trigram_lp', 'Trigram LP'),
        Regression('trigram_pmi', 'Trigram PMI')]:
    TASKS[_task.name] = _task
