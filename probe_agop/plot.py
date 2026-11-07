"""The figure of one probe at one layer: OE, CE and decodability over pretraining."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REAL, CONTROL, CE, SEL = '#2C6BAC', '#555555', '#2E8B57', '#B03A2E'


def _axis(ax, steps, ylabel):
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ['1', '10', '100', '1k', '10k', '100k'])
    ax.set_xlim(steps[0] * 0.8, steps[-1] * 1.25)
    ax.set_xlabel('Pretraining step')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def plot_layer(summary, path):
    s = summary['curves']
    steps = np.array(s['steps'])
    ev = summary['events']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    ax = axes[0]
    for seeds, mean, color, name in [(s['oe_real_seeds'], s['oe_real'], REAL, 'real probe'),
                                     (s['oe_control_seeds'], s['oe_control'], CONTROL, 'control probe')]:
        for row in np.array(seeds).T:
            ax.plot(steps, row, color=color, lw=0.6, alpha=0.35)
        ax.plot(steps, mean, color=color, lw=1.8, label=f'{name} (mean of 3 seeds)')
    ax.axhline(0, color='#999999', lw=0.8)
    _axis(ax, steps, 'OE (bits)')
    ax.set_title('Optimization Engagement')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    ax = axes[1]
    sigma = np.array(s['sigma0'])
    ax.fill_between(steps, -sigma, sigma, color='#BBBBBB', alpha=0.45, lw=0, label=r'$\pm\sigma_0$ (null noise)')
    ax.plot(steps, s['ce'], color=CE, lw=1.8, label='CE = OE(real) - OE(control)')
    ax.axhline(0, color='#999999', lw=0.8)
    if ev['t_E'] is not None:
        ax.axvline(ev['t_E'], color=CE, ls='--', lw=1, label=f"$t^E$ = {ev['t_E']:.0f}")
    _axis(ax, steps, 'CE (bits)')
    ax.set_title('Concept Energy')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(steps, s['acc_real'], color=REAL, lw=1.4, label='real probe, real labels')
    ax.plot(steps, s['acc_control'], color=CONTROL, lw=1.4, label='control probe, control labels')
    ax.plot(steps, s['selectivity'], color=SEL, lw=1.8, label='selectivity = difference')
    ax.axvline(ev['t_D'], color=SEL, ls='--', lw=1, label=f"$t^D$ = {ev['t_D']:.0f}")
    _axis(ax, steps, 'Test accuracy')
    ax.set_title('Decodability')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)

    lag = 'no energy event' if ev['lag'] is None else rf"$\ln(t^E/t^D)$ = {ev['lag']:+.2f} ({ev['relation']})"
    fig.suptitle(f"{summary['config']['model']}, layer {summary['config']['layer']}, {summary['task']}: {lag}", fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), dpi=200, facecolor='white')
    plt.close(fig)
