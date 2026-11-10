"""Optimization Engagement (OE), Concept Energy (CE) and the null noise scale sigma_0.

OE of a probe at one layer and checkpoint: let U be the top-k right singular vectors of the class-centered probe weights
W (k = n_classes - 1; for a one-output regression probe, k = 1 and U is its normalized weight vector) and G the AGOP of
the same layer and checkpoint. Then

    OE = log2( (tr(U^T G U) / tr(G)) / (k / d) ),

the log2 enrichment of the trace share captured by the probe subspace over a random k-dimensional subspace (0 = chance).
CE = OE(real probe) - OE(control probe), each averaged over its seeds.
"""
import math

import numpy as np


def probe_subspace(W, k):
    """Top-k right singular vectors of the class-centered weights W [n_classes, d], as columns [d, k]."""
    centered = W - W.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:k].T


def probe_direction(w):
    """The unit direction of a one-output (regression) probe w [1, d] or [d], as a column [d, 1] (k = 1)."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return (w / np.linalg.norm(w))[:, None]


def optimization_engagement(U, G):
    d, k = G.shape[0], U.shape[1]
    return float(np.log2(d * np.trace(U.T @ G @ U) / (k * np.trace(G))))


def participation_ratio(G):
    return float(np.trace(G) ** 2 / np.sum(G * G))   # tr(G)^2 / tr(G^2) for symmetric G


def sigma0(G, k):
    """Standard deviation of OE for a uniformly random k-dimensional subspace (paper appendix, eq. sigma0):
    Var(g) = 2d/(d+2) * (1/PR - 1/d) for the gain g of one random direction, sigma_0 = sqrt(Var(g)/k) / ln 2."""
    d = G.shape[0]
    var_g = 2 * d / (d + 2) * (1 / participation_ratio(G) - 1 / d)
    return math.sqrt(var_g / k) / math.log(2)
