"""Reading the data of the paper's tasks.

* UD English EWT v2.17: POS, dependency relation, dependency length (head direction and distance), dependency depth.
* UNER English EWT: named entities.
* STREUSLE 5.0: noun and verb supersenses.
* WikiText-103: the 1,000-line AGOP corpus, and the token positions of the n-gram tasks.

Every file is checked against the SHA-256 of the file used in the paper, so a different version fails instead of silently
giving other numbers. All three word-level datasets use the sentence filter of the paper: at least 4 tokens, at least 2
tokens containing a letter, and no URL-like token.

    python -m probe_agop.data DATA_DIR      checks every file and the derived dataset sizes
"""
import hashlib
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


URL_RE = re.compile(r"(?:^https?://|^www\.|://|\.com(?:/|$)|\.org(?:/|$)|\.net(?:/|$)|\.edu(?:/|$))", re.IGNORECASE)

FILES = {
    'ud_ewt/en_ewt-ud-train.conllu': '6fe5692f2e02198262483daba0626efe346dd19f0ce6c3ee7728b08af9b90a9c',
    'ud_ewt/en_ewt-ud-test.conllu': 'b837531faddd54f5c89cf5259cfe5306b075556a9ff19b2271393df29c6b02f8',
    'uner_ewt/en_ewt-ud-train.iob2': '647b296624433d2b0f304f81974404fb1d5ed1520cc05649b4646f5777a5497d',
    'uner_ewt/en_ewt-ud-test.iob2': '821ad826d5f503ac03b4edc3fbc2174f21fbb005ec25863caf2051c434616d6e',
    'streusle/streusle.ud_train.conllu': 'b47d61d64280ff141c411de46ce8e3102d3c3e781c685c95274a339d14418f7f',
    'streusle/streusle.ud_test.conllu': '4f8a9bbe7e99743af67711cdc332a8692fdf7fceeee7ae0b3d697f8050fb74d8',
    'wikitext-103-v1/test-00000-of-00001.parquet': 'abdfc9f83b1103b502924072460d4c92f277c9b49c313cef3e48cfcf7428e125',
    'wikitext-103-v1/train-00000-of-00002.parquet': 'c2ecca8c3250e79518e45d125f3a9a757d8014f6b2d8435c602be87c1f79ec3b',
    'wikitext-103-v1/train-00001-of-00002.parquet': '720f2503551f33c25bb822aad74d699fee4d5331a7373d0c262f1bfb01354fcf',
    'wikitext-103-v1/validation-00000-of-00001.parquet': 'a586125adab06f115018c43507ac267ea70850ce6218cbb96e08bb3b4db0899b',
}

# AGOP corpus of the paper: 1,000 lines drawn with random.Random(42) from the filtered lines of the four WikiText-103
# parquet files (read in sorted file-name order). Fingerprint = first 16 hex digits of the SHA-256 of the concatenated lines.
AGOP_CORPUS_SIZE, AGOP_CORPUS_SEED, AGOP_CORPUS_FINGERPRINT = 1000, 42, '835d783f72bdc00d'
_checked = set()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def data_file(root, name):
    """Path of a data file, after checking that it is the file used in the paper."""
    path = Path(root)/name
    if path not in _checked:
        if not path.exists():
            raise FileNotFoundError(f'{path} is missing; run bash scripts/download_data.sh')
        if sha256(path) != FILES[name]:
            raise ValueError(f'{path} differs from the file used in the paper; run bash scripts/download_data.sh')
        _checked.add(path)
    return path


def keep_sentence(tokens):
    return (len(tokens) >= 4 and sum(any(c.isalpha() for c in w) for w in tokens) >= 2
            and not any(URL_RE.search(w) for w in tokens))


@dataclass
class Words:
    """Annotated words in their sentence context.

    sentences       the tokens of each sentence; whole sentences are the model input
    sentence_index  for each word, the index of its sentence
    position        for each word, its position in that sentence
    forms           the word forms (the type-matched control labels each form)
    tags            per-word annotations, e.g. tags['upos'][i]
    """
    sentences: list = field(default_factory=list)
    sentence_index: list = field(default_factory=list)
    position: list = field(default_factory=list)
    forms: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)

    def add_word(self, position, form, **tags):
        """Adds a word of the last added sentence."""
        self.sentence_index.append(len(self.sentences) - 1)
        self.position.append(position)
        self.forms.append(form)
        for key, value in tags.items():
            self.tags.setdefault(key, []).append(value)


def read_conllu(path):
    """Sentences as lists of token dicts (id, form, upos, head, deprel, misc). Multiword-token ranges and empty nodes
    are skipped; every other line must have the 10 CoNLL-U columns."""
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
            if len(cols) != 10:
                raise ValueError(f'Malformed CoNLL-U line in {path}: {raw!r}')
            if '-' in cols[0] or '.' in cols[0]:
                continue
            current.append(dict(id=int(cols[0]), form=cols[1], upos=cols[3], head=int(cols[6]), deprel=cols[7],
                                misc=cols[9]))
    if current:
        sentences.append(current)
    return sentences


def depths(sentence):
    """Number of arcs from each token up to the root (the root word has depth 0)."""
    head = {t['id']: t['head'] for t in sentence}
    out = {}
    for t in sentence:
        depth, node, seen = 0, t['id'], {t['id']}
        while head[node] != 0:
            node = head[node]
            if node in seen:
                raise ValueError(f'Cycle in the dependency tree of {[x["form"] for x in sentence]}')
            seen.add(node)
            depth += 1
        out[t['id']] = depth
    return out


def arc_label(i, head):
    """Direction and distance of the head: ROOT, or {LEFT,RIGHT}_{1,2,3_5,6P} (LEFT = the head precedes the word)."""
    if head == 0:
        return 'ROOT'
    distance = abs(i - head)
    bucket = '1' if distance == 1 else '2' if distance == 2 else '3_5' if distance <= 5 else '6P'
    return f"{'LEFT' if head < i else 'RIGHT'}_{bucket}"


def ud_ewt_words(root, split):
    """Every non-punctuation word of the kept UD EWT sentences, with its UPOS tag, depth, head arc and coarse relation
    (subtype stripped, nmod:poss -> nmod). Sentences are fed whole, punctuation included."""
    words = Words()
    for sentence in read_conllu(data_file(root, f'ud_ewt/en_ewt-ud-{split}.conllu')):
        forms = [t['form'] for t in sentence]
        if not keep_sentence(forms):
            continue
        kept = [t for t in sentence if t['upos'] != 'PUNCT']
        if not kept:
            continue
        depth = depths(sentence)
        words.sentences.append(forms)
        for t in kept:
            words.add_word(t['id'] - 1, t['form'], upos=t['upos'], depth=depth[t['id']],
                      arc=arc_label(t['id'], t['head']), relation=t['deprel'].split(':', 1)[0])
    return words


def read_iob2(path):
    """Sentences as lists of (token, coarse tag); B-X and I-X both become X."""
    sentences, current = [], []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) != 5:
                raise ValueError(f'Malformed IOB2 line in {path}: {raw!r}')
            tag = cols[2].strip()
            current.append((cols[1], tag if tag == 'O' else tag.split('-', 1)[1]))
    if current:
        sentences.append(current)
    return sentences


def uner_words(root, split):
    """Every token of the kept UNER English EWT sentences with its entity class (O, PER, ORG, LOC, ...)."""
    words = Words()
    for sentence in read_iob2(data_file(root, f'uner_ewt/en_ewt-ud-{split}.iob2')):
        tokens = [w for w, _ in sentence]
        if not keep_sentence(tokens):
            continue
        words.sentences.append(tokens)
        for i, (w, tag) in enumerate(sentence):
            words.add_word(i, w, entity=tag)
    return words


SUPERSENSES = {
    'n.': ['n.ACT', 'n.ANIMAL', 'n.ARTIFACT', 'n.ATTRIBUTE', 'n.BODY', 'n.COGNITION', 'n.COMMUNICATION', 'n.EVENT', 'n.FEELING',
           'n.FOOD', 'n.GROUP', 'n.LOCATION', 'n.MOTIVE', 'n.NATURALOBJECT', 'n.OTHER', 'n.PERSON', 'n.PHENOMENON', 'n.PLANT',
           'n.POSSESSION', 'n.PROCESS', 'n.QUANTITY', 'n.RELATION', 'n.SHAPE', 'n.STATE', 'n.SUBSTANCE', 'n.TIME'],
    'v.': ['v.body', 'v.change', 'v.cognition', 'v.communication', 'v.competition', 'v.consumption', 'v.contact',
           'v.creation', 'v.emotion', 'v.motion', 'v.perception', 'v.possession', 'v.social', 'v.stative']}
SUPERSENSE_RE = re.compile(r'Supersense=([nv]\.\w+)')
MWE_LENGTH_RE = re.compile(r'MWELen=(\d+)')


def streusle_words(root, split, prefix):
    """Single-word expressions of the kept STREUSLE sentences with one of the 26 noun (`prefix` 'n.') or 14 verb ('v.')
    supersenses, with that supersense. Only sentences with at least one such word are kept."""
    words = Words()
    for sentence in read_conllu(data_file(root, f'streusle/streusle.ud_{split}.conllu')):
        forms = [t['form'] for t in sentence]
        if not keep_sentence(forms):
            continue
        kept = []
        for t in sentence:
            sense, length = SUPERSENSE_RE.search(t['misc']), MWE_LENGTH_RE.search(t['misc'])
            if sense and sense.group(1) in SUPERSENSES[prefix] and not (length and int(length.group(1)) > 1):
                kept.append((t['id'] - 1, t['form'], sense.group(1)))
        if not kept:
            continue
        words.sentences.append(forms)
        for position, form, sense in kept:
            words.add_word(position, form, supersense=sense)
    return words


def wikitext_lines(root):
    """Filtered, whitespace-normalized lines of all WikiText-103 parquet files, in sorted file-name order."""
    import pyarrow.parquet as pq
    lines = []
    for name in sorted(n for n in FILES if n.startswith('wikitext-103-v1/')):
        table = pq.read_table(data_file(root, name), columns=['text'])
        for cell in table.column('text').to_pylist():
            if cell is None:
                continue
            for line in cell.splitlines():
                tokens = line.split()
                if tokens and keep_sentence(tokens):
                    lines.append(' '.join(tokens))
    return lines


def agop_corpus(root):
    pool = wikitext_lines(root)
    chosen = sorted(random.Random(AGOP_CORPUS_SEED).sample(list(range(len(pool))), AGOP_CORPUS_SIZE))
    lines = [pool[i] for i in chosen]
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode('utf-8'))
    if h.hexdigest()[:16] != AGOP_CORPUS_FINGERPRINT:
        raise ValueError(f'AGOP corpus fingerprint {h.hexdigest()[:16]} != {AGOP_CORPUS_FINGERPRINT}')
    return lines


if __name__ == '__main__':
    from . import tasks
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'data')
    for name in FILES:
        data_file(root, name)
    agop_corpus(root)
    for name, task in tasks.TASKS.items():
        task.check_sizes(root)
    print(f'All data in {root} matches the paper (file hashes, AGOP corpus fingerprint {AGOP_CORPUS_FINGERPRINT}, '
          f'dataset sizes of all {len(tasks.TASKS)} tasks).')
