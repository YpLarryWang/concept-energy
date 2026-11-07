"""Type-matched control labels (the control task of the paper).

Every word type gets one fixed random label, drawn from the class distribution of the real training labels. The control
probe can therefore memorise word identity but learns nothing about the concept. The label of a word is a deterministic
function of (word form, seed): a SHA-256 hash of both is turned into a uniform number in [0, 1) and mapped through the
cumulative class shares. Train and test words of the same type share the label.
"""
import hashlib

import numpy as np

# Hash domain of the paper runs. It names POS because the rule was first written for POS; all classification tasks use it.
DOMAIN = b'fv-agop\x00pos-type-matched-marginal-label\x00v1\x00'
SEEDS = (42, 43, 44)


def type_matched_label(word, seed, cumulative_shares):
    form = word.encode('utf-8', errors='strict')
    framed = DOMAIN + int(seed).to_bytes(8, 'big', signed=True) + len(form).to_bytes(8, 'big', signed=False) + form
    u = int.from_bytes(hashlib.sha256(framed).digest(), 'big') / float(1 << 256)
    return int(np.searchsorted(cumulative_shares, u, side='right'))


def type_matched_labels(train_words, train_labels, test_words, n_classes, seed):
    shares = np.bincount(train_labels, minlength=n_classes) / len(train_labels)
    cumulative = np.cumsum(shares)
    cumulative[-1] = 1.0
    train = np.array([type_matched_label(w, seed, cumulative) for w in train_words], dtype=np.int64)
    test = np.array([type_matched_label(w, seed, cumulative) for w in test_words], dtype=np.int64)
    return train, test
