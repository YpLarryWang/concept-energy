"""Text data used by the POS example.

* UD English EWT: every non-punctuation word of the kept sentences, labelled with its UPOS tag.
* WikiText-103: the 1,000-line corpus on which the AGOP is computed.

Both use the sentence filter of the paper: at least 4 tokens, at least 2 tokens containing a letter, and no URL-like token.
"""
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL_RE = re.compile(r"(?:^https?://|^www\.|://|\.com(?:/|$)|\.org(?:/|$)|\.net(?:/|$)|\.edu(?:/|$))", re.IGNORECASE)

# Sizes of the POS data in the paper (UD English EWT v2.17); a different treebank version fails here instead of silently
# giving other numbers.
POS_SIZES = {'n_classes': 16, 'train': 176838, 'test': 20706}

# AGOP corpus of the paper: 1,000 lines drawn with random.Random(42) from the filtered lines of the four WikiText-103
# parquet files (test, train x2, validation; read in sorted file-name order). Fingerprint = first 16 hex digits of the
# SHA-256 of the concatenated lines.
AGOP_CORPUS_SIZE, AGOP_CORPUS_SEED, AGOP_CORPUS_FINGERPRINT = 1000, 42, '835d783f72bdc00d'


def keep_sentence(words):
    return (len(words) >= 4 and sum(any(c.isalpha() for c in w) for w in words) >= 2
            and not any(URL_RE.search(w) for w in words))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def read_conllu(path):
    """Sentences as lists of (form, upos). Multiword-token lines and empty nodes are skipped."""
    sentences, current = [], []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) < 8:
                raise ValueError(f'Malformed CoNLL-U line in {path}: {raw!r}')
            if '-' in cols[0] or '.' in cols[0]:
                continue
            if int(cols[6]) < 0:
                raise ValueError(f'Negative HEAD in {path}: {raw!r}')
            current.append((cols[1], cols[3]))
    if current:
        sentences.append(current)
    return sentences


@dataclass
class WordData:
    """Labelled words and their sentence context.

    sentences      every word of each kept sentence (punctuation included); this is the model input
    sentence_index for each labelled word, the index of its sentence
    position       for each labelled word, its position in that sentence
    words          the word forms (used by the type-matched control)
    labels         class indices
    """
    sentences: list
    sentence_index: list
    position: list
    words: list
    labels: np.ndarray


def load_pos(path, tag_names=None):
    """Non-punctuation words of the kept sentences with their UPOS tags. Tags are indexed in alphabetical order of the
    training tags; pass the training tag list when loading the test split."""
    sentences, sentence_index, position, words, tags = [], [], [], [], []
    for sentence in read_conllu(path):
        forms = [form for form, _ in sentence]
        if not keep_sentence(forms):
            continue
        kept = [(i, form, upos) for i, (form, upos) in enumerate(sentence) if upos != 'PUNCT' and form]
        if not kept:
            continue
        for i, form, upos in kept:
            sentence_index.append(len(sentences))
            position.append(i)
            words.append(form)
            tags.append(upos)
        sentences.append(forms)
    if tag_names is None:
        tag_names = sorted(set(tags))
    index = {tag: i for i, tag in enumerate(tag_names)}
    labels = np.array([index[tag] for tag in tags], dtype=np.int64)  # an unseen test tag raises KeyError
    return WordData(sentences, sentence_index, position, words, labels), tag_names


def load_pos_splits(ewt_dir):
    ewt_dir = Path(ewt_dir)
    train, tags = load_pos(ewt_dir/'en_ewt-ud-train.conllu')
    test, _ = load_pos(ewt_dir/'en_ewt-ud-test.conllu', tags)
    got = {'n_classes': len(tags), 'train': len(train.words), 'test': len(test.words)}
    if got != POS_SIZES:
        raise ValueError(f'POS data sizes {got} differ from the paper {POS_SIZES}; use UD English EWT v2.17')
    return train, test, tags


def wikitext_lines(wikitext_dir):
    """Filtered, whitespace-normalised lines of all WikiText-103 parquet files, in sorted file-name order."""
    import pyarrow.parquet as pq
    files = sorted(Path(wikitext_dir).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No parquet files in {wikitext_dir}')
    lines = []
    for path in files:
        for cell in pq.read_table(path, columns=['text']).column('text').to_pylist():
            if cell is None:
                continue
            for line in cell.splitlines():
                tokens = line.split()
                if tokens and keep_sentence(tokens):
                    lines.append(' '.join(tokens))
    return lines


def agop_corpus(wikitext_dir):
    pool = wikitext_lines(wikitext_dir)
    chosen = sorted(random.Random(AGOP_CORPUS_SEED).sample(list(range(len(pool))), AGOP_CORPUS_SIZE))
    lines = [pool[i] for i in chosen]
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode('utf-8'))
    if h.hexdigest()[:16] != AGOP_CORPUS_FINGERPRINT:
        raise ValueError(f'AGOP corpus fingerprint {h.hexdigest()[:16]} != {AGOP_CORPUS_FINGERPRINT}; '
                         f'{wikitext_dir} must hold the four parquet files of Salesforce/wikitext, wikitext-103-v1')
    return lines
