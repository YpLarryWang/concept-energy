"""Event times on the checkpoint axis (paper Sec. 4, "Event times").

All times live on a log-step axis.
  t^D  decodability event: centre (geometric mean of the two ends) of the checkpoint interval where decodability rises
       fastest per unit log-step.
  t^E  energy event: the earliest qualifying interior peak of [CE]_+ = max(CE, 0).
  lag  ln(t^E / t^D); |lag| < ln2/2 counts as the same time, lag <= -ln2/2 as energy earlier, lag >= ln2/2 as
       decodability earlier.

A peak of [CE]_+ at checkpoint i is an interior local maximum (strictly above both neighbours). Its valleys are found by
walking left and right while the curve keeps falling. With r = max - min of the whole curve, the peak qualifies when
    (peak - higher valley) / r >= 0.05   and
    ((peak - higher valley) / r >= 0.25  or  (peak - lower valley) / r >= 0.75).
"""
import math

import numpy as np

HALF_STEP = math.log(2) / 2
MIN_PROMINENCE, HIGH_VALLEY_PROMINENCE, LOW_VALLEY_PROMINENCE = 0.05, 0.25, 0.75


def decodability_event(steps, decodability):
    log_steps = np.log(np.asarray(steps, dtype=float))
    slopes = np.diff(np.asarray(decodability, dtype=float)) / np.diff(log_steps)
    i = int(np.argmax(slopes))
    return float(math.exp(0.5 * (log_steps[i] + log_steps[i + 1])))


def energy_event(steps, ce):
    """Earliest qualifying peak of [CE]_+, or None when the curve has none."""
    curve = np.maximum(np.asarray(ce, dtype=float), 0.0)
    span = curve.max() - curve.min()
    if span <= 0:
        return None
    for i in range(1, len(curve) - 1):
        if not curve[i - 1] < curve[i] > curve[i + 1]:
            continue
        left = i - 1
        while left > 0 and curve[left - 1] <= curve[left]:
            left -= 1
        right = i + 1
        while right < len(curve) - 1 and curve[right + 1] <= curve[right]:
            right += 1
        high_valley = (curve[i] - max(curve[left], curve[right])) / span
        low_valley = (curve[i] - min(curve[left], curve[right])) / span
        if high_valley >= MIN_PROMINENCE and (high_valley >= HIGH_VALLEY_PROMINENCE or low_valley >= LOW_VALLEY_PROMINENCE):
            return float(steps[i])
    return None


def relation(lag):
    if lag <= -HALF_STEP:
        return 'energy earlier'
    if lag >= HALF_STEP:
        return 'decodability earlier'
    return 'same time'


def layer_events(steps, ce, decodability):
    t_d, t_e = decodability_event(steps, decodability), energy_event(steps, ce)
    lag = None if t_e is None else math.log(t_e / t_d)
    return dict(t_D=t_d, t_E=t_e, lag=lag, relation=None if lag is None else relation(lag),
                weight=float(np.maximum(ce, 0).max()))


def task_events(per_layer):
    """Task-level event times from several layers: geometric means over the layers with an energy event, weighted by
    each layer's max [CE]_+ (paper eq. aggregated-event-time)."""
    valid = [e for e in per_layer if e['t_E'] is not None and e['weight'] > 0]
    if not valid:
        return None
    w = np.array([e['weight'] for e in valid])
    t_d = math.exp(np.average([math.log(e['t_D']) for e in valid], weights=w))
    t_e = math.exp(np.average([math.log(e['t_E']) for e in valid], weights=w))
    return dict(t_D=t_d, t_E=t_e, lag=math.log(t_e / t_d), relation=relation(math.log(t_e / t_d)), n_layers=len(valid))
