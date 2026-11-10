"""CPU-only checks of the metric definitions (numpy; no model or GPU needed): python -m pytest tests"""
import json
import math
from pathlib import Path

import numpy as np

from probe_agop import controls, energy, events

STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
REFERENCE = Path(__file__).resolve().parents[1]/'reference'


def test_oe_is_zero_for_isotropic_agop_and_positive_on_top_eigenvectors():
    rng = np.random.default_rng(0)
    d, k = 64, 5
    W = rng.normal(size=(k + 1, d))
    assert abs(energy.optimization_engagement(energy.probe_subspace(W, k), np.eye(d))) < 1e-12
    G = np.diag(np.r_[np.full(k, 10.0), np.ones(d - k)])
    U = np.eye(d)[:, :k]
    assert math.isclose(energy.optimization_engagement(U, G), math.log2(d * 10 / (10 * k + d - k)))


def test_probe_subspace_ignores_a_shift_shared_by_all_classes():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(4, 32))
    U1 = energy.probe_subspace(W, 3)
    U2 = energy.probe_subspace(W + rng.normal(size=(1, 32)), 3)
    assert np.allclose(U1 @ U1.T, U2 @ U2.T)


def test_sigma0_matches_random_subspaces():
    rng = np.random.default_rng(2)
    d, k = 256, 15
    G = np.diag(rng.gamma(0.3, size=d))
    oe = []
    for _ in range(4000):
        Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
        oe.append(energy.optimization_engagement(Q, G))
    assert abs(np.std(oe) / energy.sigma0(G, k) - 1) < 0.1


def test_control_labels_follow_word_types_and_class_shares():
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat'] * 500 + [f'w{i}' for i in range(20000)]
    labels = np.r_[np.zeros(13000, int), np.ones(10000, int)]
    train, test = controls.control_labels(words, labels, ['the', 'cat'], 2, seed=42)
    first = {}
    for w, y in zip(words, train):
        assert first.setdefault(w, y) == y
    assert test.tolist() == [first['the'], first['cat']]
    assert abs(np.mean(train[3000:] == 0) - 13000 / 23000) < 0.02


def test_events_on_simple_curves():
    ce = np.zeros(19)
    ce[6:9] = [0.5, 1.0, 0.4]          # peak at step 128
    ce[15] = 0.2                       # small late bump (prominence below the gate)
    decod = np.r_[np.zeros(9), np.ones(10)]   # rises between steps 256 and 512
    ev = events.layer_events(STEPS, ce, decod)
    assert ev['t_E'] == 128 and math.isclose(ev['t_D'], math.sqrt(256 * 512))
    assert ev['relation'] == 'energy earlier'
    assert events.energy_event(STEPS, -np.ones(19)) is None


def test_events_reproduce_the_paper_for_every_task_and_model():
    """Per-layer events and task-level event times computed from the reference values (OE and scores, rounded to 7
    decimals) equal the paper's."""
    files = sorted(REFERENCE.glob('*/*.json'))
    assert len(files) == 24
    for path in files:
        ref = json.loads(path.read_text())
        per_layer = []
        for layer, r in ref['layers'].items():
            ce = np.mean(r['oe_real'], 1) - np.mean(r['oe_control'], 1)
            selectivity = np.mean(r['score_real'], 1) - np.mean(r['score_control'], 1)
            ev = events.layer_events(ref['steps'], ce, selectivity)
            assert math.isclose(ev['t_D'], r['events']['t_D']) and ev['t_E'] == r['events']['t_E'], (path, layer)
            per_layer.append(ev)
        task = events.task_events(per_layer)
        paper = ref['task_events']
        assert math.isclose(task['t_D'], paper['t_D'], rel_tol=1e-6) and math.isclose(task['t_E'], paper['t_E'], rel_tol=1e-6)
        assert task['relation'] == paper['relation'], path
