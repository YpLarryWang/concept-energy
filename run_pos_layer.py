"""POS probe vs AGOP over pretraining, for one layer of one Pythia model.

For each checkpoint: compute the AGOP at the layer, extract the layer's word representations on UD English EWT, train
3 real POS probes and 3 type-matched control probes, and measure each probe's Optimization Engagement (OE) with the AGOP.
Then: CE = OE(real) - OE(control), decodability = selectivity (real accuracy - control accuracy), the null noise scale
sigma_0, the event times t^D and t^E and their lag, and a figure of all curves.

    python run_pos_layer.py --layer 8 --out runs/pythia-410m_pos_layer8

Per-checkpoint results are written as they finish; rerunning the same command resumes. For Pythia-410M the result is
compared with the paper's values for the same layer (reference/pythia-410m_pos.json).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe_agop import agop, controls, data, energy, events, probe
from probe_agop import model as lm
from probe_agop.plot import plot_layer

CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
REAL_SEEDS = (1001, 1002, 1003)            # probe seeds of the real probes
CONTROL_INIT_OFFSET = 4000                 # control probe with label seed s uses probe seed 4000 + s
SCHEMA = 'probe-agop/pos-layer/v1'
REPO = Path(__file__).resolve().parent


def run_config(args, ewt_dir):
    return dict(schema=SCHEMA, task='pos', model=args.model, layer=args.layer,
                ewt_sha256={s: data.sha256(ewt_dir/f'en_ewt-ud-{s}.conllu') for s in ('train', 'test')},
                agop=dict(corpus_fingerprint=data.AGOP_CORPUS_FINGERPRINT, tokens_per_sequence=agop.TOKENS_PER_SEQUENCE,
                          max_length=agop.MAX_LENGTH, position_seed=agop.POSITION_SEED, objective='next-token loss',
                          dtype='float32'),
                representations=dict(dtype='float16', site='block output, last sub-word token of each word'),
                probe=dict(lr=probe.LR, weight_decay=probe.WEIGHT_DECAY, batch_size=probe.BATCH_SIZE, epochs=probe.EPOCHS,
                           inputs='centred with the training mean', real_seeds=list(REAL_SEEDS),
                           control='type_matched', control_label_seeds=list(controls.SEEDS),
                           control_init_offset=CONTROL_INIT_OFFSET))


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def run_checkpoint(step, args, cfg, splits, corpus, tokenizer, out):
    (train, test, tags), (control_train, control_test) = splits
    n_classes, k = len(tags), len(tags) - 1
    t0 = time.time()
    on_gpu = args.device.startswith('cuda')
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    # AGOP in float32 (late checkpoints are sensitive to float16 rounding in the backward pass); word representations in
    # float16, as in the paper. The float16 model is loaded afresh: model.half() would also round the rotary-embedding
    # frequencies, which from_pretrained keeps in float32.
    model = lm.load_model(args.model, step, args.device, torch.float32)
    G = agop.compute_agop(model, tokenizer, args.layer, corpus, batch_size=args.agop_batch_size)
    np.save(out/'agop'/f'step{step}.npy', G)
    del model
    t_agop = time.time()
    model = lm.load_model(args.model, step, args.device, torch.float16)
    X_train = lm.word_representations(model, tokenizer, train, args.layer)
    X_test = lm.word_representations(model, tokenizer, test, args.layer)
    del model
    torch.cuda.empty_cache()
    X_train, X_test = probe.centre(X_train, X_test)
    t_reps = time.time()
    y_train, y_test = torch.from_numpy(train.labels), torch.from_numpy(test.labels)
    runs = [('real', s, s, y_train, y_test) for s in REAL_SEEDS]
    runs += [('control', s, CONTROL_INIT_OFFSET + s, torch.from_numpy(control_train[s]), torch.from_numpy(control_test[s]))
             for s in controls.SEEDS]
    probes = []
    for kind, label_seed, init_seed, y_tr, y_te in runs:
        p = probe.train_probe(X_train, y_tr, n_classes, init_seed, args.device)
        U = energy.probe_subspace(p.weight.detach().double().cpu().numpy(), k)
        probes.append(dict(kind=kind, label_seed=label_seed, init_seed=init_seed,
                           oe=energy.optimization_engagement(U, G), test_acc=probe.accuracy(p, X_test, y_te)))
    record = dict(step=step, config_fingerprint=fingerprint(cfg), k=k, sigma0=energy.sigma0(G, k),
                  participation_ratio=energy.participation_ratio(G), probes=probes,
                  seconds=dict(agop=t_agop - t0, representations=t_reps - t_agop, probes=time.time() - t_reps),
                  peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 2**30 if on_gpu else None)
    path = out/'checkpoints'/f'step{step}.json'
    path.write_text(json.dumps(record, indent=1))
    real = [p['oe'] for p in probes if p['kind'] == 'real']
    ctrl = [p['oe'] for p in probes if p['kind'] == 'control']
    print(f'step {step:>6}: OE real {np.mean(real):+.3f}  OE control {np.mean(ctrl):+.3f}  '
          f'CE {np.mean(real) - np.mean(ctrl):+.3f}  ({time.time() - t0:.0f} s)', flush=True)
    return record


def load_record(path, cfg):
    record = json.loads(path.read_text())
    if record['config_fingerprint'] != fingerprint(cfg):
        raise RuntimeError(f'{path} was written with another configuration; use a new --out directory')
    return record


def summarise(records, cfg):
    steps = [r['step'] for r in records]

    def seeds(kind, key):
        return np.array([[p[key] for p in r['probes'] if p['kind'] == kind] for r in records])

    oe_real, oe_ctrl = seeds('real', 'oe'), seeds('control', 'oe')
    acc_real, acc_ctrl = seeds('real', 'test_acc'), seeds('control', 'test_acc')
    ce = oe_real.mean(1) - oe_ctrl.mean(1)
    sel = acc_real.mean(1) - acc_ctrl.mean(1)
    sigma0 = np.array([r['sigma0'] for r in records])
    curves = dict(steps=steps, oe_real=oe_real.mean(1), oe_control=oe_ctrl.mean(1), ce=ce, sigma0=sigma0,
                  acc_real=acc_real.mean(1), acc_control=acc_ctrl.mean(1), selectivity=sel,
                  oe_real_seeds=oe_real, oe_control_seeds=oe_ctrl, acc_real_seeds=acc_real, acc_control_seeds=acc_ctrl)
    curves = {key: np.asarray(v).tolist() for key, v in curves.items()}
    late = np.array(steps) >= 128
    stats = dict(k=records[0]['k'], median_sigma0=float(np.median(sigma0)),
                 median_ce_over_sigma0_from_step128=float(np.median(ce[late] / sigma0[late])) if late.any() else None,
                 ce_positive_share=float(np.mean(ce > 0)),
                 seed_sd_over_sigma0=dict(real=float(np.median(oe_real.std(1, ddof=1) / sigma0)),
                                          control=float(np.median(oe_ctrl.std(1, ddof=1) / sigma0))))
    return dict(task='POS (16 UPOS classes)', config=cfg, curves=curves, events=events.layer_events(steps, ce, sel),
                stats=stats)


def print_summary(summary):
    c, ev, st = summary['curves'], summary['events'], summary['stats']
    print(f"\n{'step':>7} {'OE real':>8} {'OE ctrl':>8} {'CE':>7} {'sigma0':>7} {'acc real':>9} {'acc ctrl':>9} {'select.':>8}")
    for i, step in enumerate(c['steps']):
        print(f"{step:>7} {c['oe_real'][i]:>+8.3f} {c['oe_control'][i]:>+8.3f} {c['ce'][i]:>+7.3f} {c['sigma0'][i]:>7.3f} "
              f"{c['acc_real'][i]:>9.4f} {c['acc_control'][i]:>9.4f} {c['selectivity'][i]:>+8.4f}")
    t_e = 'none' if ev['t_E'] is None else f"{ev['t_E']:.0f}"
    lag = 'n/a' if ev['lag'] is None else f"{ev['lag']:+.2f} ({ev['relation']})"
    print(f"\nt^D = {ev['t_D']:.0f}   t^E = {t_e}   lag ln(t^E/t^D) = {lag}")
    if len(c['steps']) < len(CHECKPOINTS):
        print('(event times are only meaningful on the full checkpoint grid)')
    snr = st['median_ce_over_sigma0_from_step128']
    print(f"sigma0 (k = {st['k']}): median {st['median_sigma0']:.3f} bits; median CE/sigma0 from step 128 on: "
          f"{'n/a' if snr is None else f'{snr:.1f}'}; seed SD of OE / sigma0: real {st['seed_sd_over_sigma0']['real']:.2f}, "
          f"control {st['seed_sd_over_sigma0']['control']:.2f}")


def compare_with_paper(summary, reference_path):
    """Differences from the paper's values for the same model, task and layer."""
    ref = json.loads(reference_path.read_text())['layers'][str(summary['config']['layer'])]
    c = summary['curves']
    idx = [ref['steps'].index(s) for s in c['steps']]
    out = {}
    for key in ('oe_real', 'oe_control', 'ce', 'acc_real', 'acc_control', 'selectivity'):
        diff = np.abs(np.array(c[key]) - np.array(ref[key])[idx])
        out[key] = dict(max_abs_diff=float(diff.max()), at_step=c['steps'][int(diff.argmax())])
    out['events'] = dict(paper=ref['events'], this_run=summary['events']) if len(idx) == len(ref['steps']) else None
    print('\nComparison with the paper (same layer): max |difference| over checkpoints')
    for key, v in out.items():
        if key != 'events':
            print(f"  {key:<12} {v['max_abs_diff']:.4f} (step {v['at_step']})")
    if out['events'] is not None:
        p = ref['events']
        print(f"  paper events: t^D = {p['t_D']:.0f}, t^E = {p['t_E'] if p['t_E'] is None else round(p['t_E'])}, "
              f"lag = {p['lag'] if p['lag'] is None else round(p['lag'], 2)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--out', required=True, help='output directory (created if missing; reused to resume)')
    ap.add_argument('--model', default='EleutherAI/pythia-410m')
    ap.add_argument('--data-dir', default=str(REPO/'data'), help='directory filled by scripts/download_data.sh')
    ap.add_argument('--checkpoints', default=','.join(map(str, CHECKPOINTS)),
                    help='comma-separated training steps (default: the 19 checkpoints of the paper)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--agop-batch-size', type=int, default=8)
    args = ap.parse_args()
    steps = [int(s) for s in args.checkpoints.split(',')]
    if steps != sorted(set(steps)):
        raise ValueError('--checkpoints must be increasing and distinct')

    data_dir = Path(args.data_dir)
    ewt_dir = data_dir/'ud_ewt'
    out = Path(args.out)
    (out/'checkpoints').mkdir(parents=True, exist_ok=True)
    (out/'agop').mkdir(exist_ok=True)
    cfg = run_config(args, ewt_dir)
    cfg_path = out/'config.json'
    if cfg_path.exists() and fingerprint(json.loads(cfg_path.read_text())) != fingerprint(cfg):
        raise RuntimeError(f'{out} holds a run with another configuration; use a new --out directory')
    cfg_path.write_text(json.dumps(cfg, indent=1))

    todo = [s for s in steps if not (out/'checkpoints'/f'step{s}.json').exists()]
    if todo:
        train, test, tags = data.load_pos_splits(ewt_dir)
        control = {s: controls.type_matched_labels(train.words, train.labels, test.words, len(tags), s) for s in controls.SEEDS}
        splits = ((train, test, tags), ({s: v[0] for s, v in control.items()}, {s: v[1] for s, v in control.items()}))
        corpus = data.agop_corpus(data_dir/'wikitext-103-v1')
        tokenizer = lm.load_tokenizer(args.model)
        print(f'{args.model}, layer {args.layer}: {len(todo)} checkpoint(s) to run, {len(steps) - len(todo)} done', flush=True)
        for step in todo:
            run_checkpoint(step, args, cfg, splits, corpus, tokenizer, out)
    records = [load_record(out/'checkpoints'/f'step{s}.json', cfg) for s in steps]

    summary = summarise(records, cfg)
    print_summary(summary)
    reference = REPO/'reference'/f"{args.model.split('/')[-1]}_pos.json"
    if reference.exists():
        summary['comparison_with_paper'] = compare_with_paper(summary, reference)
    (out/'summary.json').write_text(json.dumps(summary, indent=1))
    plot_layer(summary, out/'curves')
    print(f'\nWrote {out}/summary.json and {out}/curves.png')


if __name__ == '__main__':
    main()
