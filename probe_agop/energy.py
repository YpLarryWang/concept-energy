"""Optimization Engagement (OE), Concept Energy (CE) and the null noise scale sigma_0.

OE of a probe at one layer and checkpoint: let U be the top-k right singular vectors of the class-centred probe weights
W (k = n_classes - 1) and G the AGOP of the same layer and checkpoint. Then

    OE = log2( (tr(U^T G U) / tr(G)) / (k / d) ),

the log2 enrichment of the trace share captured by the probe subspace over a random k-dimensional subspace (0 = chance).
CE = OE(real probe) - OE(control probe), each averaged over its seeds.
"""
import math

import numpy as np


def probe_subspace(W, k):
    """Top-k right singular vectors of the class-centred weights W [n_classes, d], as columns [d, k]."""
    centred = W - W.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:k].T


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
