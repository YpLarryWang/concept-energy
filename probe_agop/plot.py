"""Figures: one probe at one layer over pretraining (OE, CE, decodability), and the task-level event times."""
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import write_atomic
from .events import HALF_STEP

REAL, CONTROL, CE, SEL = '#2C6BAC', '#555555', '#2E8B57', '#B03A2E'
TICKS = ([1, 10, 100, 1000, 10000, 100000], ['1', '10', '100', '1k', '10k', '100k'])


def _axis(ax, steps, ylabel):
    ax.set_xscale('log')
    ax.set_xticks(*TICKS)
    ax.set_xlim(steps[0] * 0.8, steps[-1] * 1.25)
    ax.set_xlabel('Pretraining step')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _save(fig, path):
    for ext in ('png', 'pdf'):
        write_atomic(path.with_suffix('.' + ext), lambda f: fig.savefig(f, format=ext, dpi=200, facecolor='white'))
    plt.close(fig)


def plot_layer(summary, model, path):
    s = summary['curves']
    steps = np.array(s['steps'])
    ev = summary['events']
    mark = dict(marker='o', ms=4) if len(steps) == 1 else {}     # a single checkpoint has no line to draw
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    ax = axes[0]
    for seeds, mean, color, name in [(s['oe_real_seeds'], s['oe_real'], REAL, 'real probe'),
                                     (s['oe_control_seeds'], s['oe_control'], CONTROL, 'control probe')]:
        for row in np.array(seeds).T:
            ax.plot(steps, row, color=color, lw=0.6, alpha=0.35, **mark)
        ax.plot(steps, mean, color=color, lw=1.8, label=f'{name} (mean of 3 seeds)', **mark)
    ax.axhline(0, color='#999999', lw=0.8)
    _axis(ax, steps, 'OE (bits)')
    ax.set_title('Optimization Engagement')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    ax = axes[1]
    sigma = np.array(s['sigma0'])
    if len(steps) > 1:
        ax.fill_between(steps, -sigma, sigma, color='#BBBBBB', alpha=0.45, lw=0, label=r'$\pm\sigma_0$ (null noise)')
    else:
        ax.errorbar(steps, [0], yerr=sigma, fmt='none', ecolor='#BBBBBB', elinewidth=6, label=r'$\pm\sigma_0$ (null noise)')
    ax.plot(steps, s['ce'], color=CE, lw=1.8, label='CE = OE(real) - OE(control)', **mark)
    ax.axhline(0, color='#999999', lw=0.8)
    if ev['t_E'] is not None:
        ax.axvline(ev['t_E'], color=CE, ls='--', lw=1, label=f"$t^E$ = {ev['t_E']:.0f}")
    _axis(ax, steps, 'CE (bits)')
    ax.set_title('Concept Energy')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(steps, s['score_real'], color=REAL, lw=1.4, label='real probe, real labels', **mark)
    ax.plot(steps, s['score_control'], color=CONTROL, lw=1.4, label='control probe, control labels', **mark)
    ax.plot(steps, s['selectivity'], color=SEL, lw=1.8, label='selectivity = difference', **mark)
    if ev['t_D'] is not None:
        ax.axvline(ev['t_D'], color=SEL, ls='--', lw=1, label=f"$t^D$ = {ev['t_D']:.0f}")
    _axis(ax, steps, summary['score'])
    ax.set_title('Decodability')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    lag = ('event times need more checkpoints' if len(steps) < 3 else 'no energy event' if ev['t_E'] is None else 'no decodability event' if ev['t_D'] is None else
           rf"$\ln(t^E/t^D)$ = {ev['lag']:+.2f} ({ev['relation']})")
    fig.suptitle(f"{model}, layer {summary['layer']}, {summary['title']}: {lag}", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_event_times(results, model, path):
    """Task-level t^D (x) against t^E (y), one point per task; points below the diagonal: energy before decodability.
    The band marks |ln(t^E/t^D)| < ln2/2 (same time)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    lo, hi = 10, 3e4
    grid = np.geomspace(lo, hi, 50)
    ax.fill_between(grid, grid * math.exp(-HALF_STEP), grid * math.exp(HALF_STEP), color='#DDDDDD', lw=0,
                    label='same time (within half a checkpoint step)')
    ax.plot(grid, grid, color='#888888', lw=0.8)
    for name, r in results.items():
        te = r['task_events']
        ax.scatter(te['t_D'], te['t_E'], s=28, color=CE if te['relation'] == 'energy earlier' else
                   SEL if te['relation'] == 'decodability earlier' else CONTROL, zorder=3)
        ax.annotate(r['title'], (te['t_D'], te['t_E']), textcoords='offset points', xytext=(4, 3), fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('$t^D$ (decodability event, step)')
    ax.set_ylabel('$t^E$ (energy event, step)')
    ax.set_title(f'{model}: task-level event times', fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, path)
