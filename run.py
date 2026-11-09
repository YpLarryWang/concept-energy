"""Probe vs AGOP over pretraining: one command for any task, layer and Pythia model of the paper.

For each checkpoint: compute the AGOP of the next-token loss at every requested layer (shared by all tasks), extract the
layers' representations of the task's words or tokens, train 3 real probes and 3 type-matched control probes per
(task, layer), and measure each probe's Optimization Engagement (OE) with the AGOP. Then, per (task, layer):
CE = OE(real) - OE(control), decodability = selectivity (real score - control score), the null noise scale sigma_0, the
event times t^D and t^E and their lag, and a figure. With all layers of a model, also the task-level event times (the
points of the paper's event-time figure).

    python run.py --task pos --layers 8                          # one layer of Pythia-410M, 19 checkpoints
    python run.py --model pythia-1b --task all --layers all      # everything for Pythia-1B

Results are written per (task, layer, checkpoint) as they finish; rerunning the same command resumes. Every result is
compared with the paper's values (reference/<model>/<task>.json).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from probe_agop import agop, data, energy, events, ngram, probe, tasks
from probe_agop import model as lm
from probe_agop.plot import plot_event_times, plot_layer

CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
MODELS = {'pythia-410m': 24, 'pythia-1b': 16}       # number of transformer blocks
REAL_SEEDS = (1001, 1002, 1003)                      # probe seeds of the real probes
CONTROL_SEEDS = (42, 43, 44)                         # label seeds of the type-matched controls
CONTROL_INIT_OFFSET = 4000                           # the control with label seed s uses probe seed 4000 + s
SCHEMA = 'concept-energy/v2'
# Agreement with the paper: largest allowed |difference| over checkpoints. OE and CE differ slightly because the AGOP is
# recomputed in float32 (the paper's cache mixes float32 and float16 runs); test accuracy and held-out R^2 match exactly
# on an A100, and other GPUs may round differently.
TOLERANCE = dict(oe=0.02, score=0.002)
REPO = Path(__file__).resolve().parent


def task_config(model, task):
    return dict(schema=SCHEMA, model=f'EleutherAI/{model}', task=task.name, data=task.data_fingerprint(),
                agop=dict(corpus_fingerprint=data.AGOP_CORPUS_FINGERPRINT, tokens_per_sequence=agop.TOKENS_PER_SEQUENCE,
                          max_length=agop.MAX_LENGTH, position_seed=agop.POSITION_SEED, objective='next-token loss',
                          dtype='float32'),
                representations=dict(dtype='float16', site='block output', unit=task.unit),
                probe=dict(task.recipe(), real_seeds=list(REAL_SEEDS), control='type_matched',
                           control_label_seeds=list(CONTROL_SEEDS), control_init_offset=CONTROL_INIT_OFFSET))


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def record_path(out, task, layer, step):
    return out/task/f'layer{layer}'/f'step{step}.json'


def agop_for(model_name, step, layers, out, corpus, tokenizer, args):
    """AGOP of every layer in `layers` at `step`, from out/agop/ or computed in one pass (float32 model)."""
    path = lambda layer: out/'agop'/f'step{step}'/f'layer{layer}.npy'
    missing = [layer for layer in layers if not path(layer).exists()]
    if missing:
        t0 = time.time()
        model = lm.load_model(f'EleutherAI/{model_name}', step, args.device, torch.float32)
        G = agop.compute_agops(model, tokenizer, missing, corpus, batch_size=args.agop_batch_size)
        del model
        torch.cuda.empty_cache()
        path(missing[0]).parent.mkdir(parents=True, exist_ok=True)
        for layer, g in G.items():                   # write then rename: parallel runs never read a partial file
            tmp = path(layer).with_suffix(f'.{os.getpid()}.tmp')
            with open(tmp, 'wb') as f:
                np.save(f, g)
            os.replace(tmp, path(layer))
        print(f'  AGOP of {len(missing)} layer(s): {time.time() - t0:.0f} s', flush=True)
    return {layer: np.load(path(layer)) for layer in layers}


def run_probes(task, td, X_train, X_test, G, device):
    """The 6 probes of one (task, layer, checkpoint): OE and held-out score of each."""
    X_train, X_test = probe.centre(X_train, X_test)
    runs = [('real', s, s, td.y_train, td.y_test) for s in REAL_SEEDS]
    runs += [('control', s, CONTROL_INIT_OFFSET + s, *td.controls[s]) for s in CONTROL_SEEDS]
    out = []
    for kind, label_seed, init_seed, y_train, y_test in runs:
        U, score = task.fit(td, X_train, y_train, X_test, y_test, init_seed, device)
        out.append(dict(kind=kind, label_seed=label_seed, init_seed=init_seed, oe=energy.optimization_engagement(U, G),
                        score=score))
    return out


def run_checkpoint(step, todo, args, out, cache, corpus, tokenizer):
    """todo: {task name: [layers]} still missing at this checkpoint."""
    t0 = time.time()
    on_gpu = args.device.startswith('cuda')
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    layers = sorted({layer for ls in todo.values() for layer in ls})
    G = agop_for(args.model, step, layers, out, corpus, tokenizer, args)
    model = lm.load_model(f'EleutherAI/{args.model}', step, args.device, torch.float16)
    by_key = {}
    for name in todo:
        by_key.setdefault(cache[name].key, []).append(name)
    for key, names in by_key.items():
        key_layers = sorted({layer for name in names for layer in todo[name]})
        shared = cache[names[0]]                      # tasks with one key read the same words or tokens
        for i in range(0, len(key_layers), args.layer_chunk):
            chunk = key_layers[i:i + args.layer_chunk]
            reps = shared.representations(model, tokenizer, chunk)
            for name in names:
                task, td = tasks.TASKS[name], cache[name]
                for layer in chunk:
                    if layer not in todo[name]:
                        continue
                    t1 = time.time()
                    X_train, X_test = td.select(reps, layer)
                    probes = run_probes(task, td, X_train, X_test, G[layer], args.device)
                    record = dict(step=step, layer=layer, task=name, config_fingerprint=args.fingerprints[name], k=td.k,
                                  sigma0=energy.sigma0(G[layer], td.k), participation_ratio=energy.participation_ratio(G[layer]),
                                  probes=probes, seconds=time.time() - t1)
                    path = record_path(out, name, layer, step)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record, indent=1))
                    oe = lambda kind: np.mean([p['oe'] for p in probes if p['kind'] == kind])
                    print(f'  {name:<16} layer {layer:>2}: OE real {oe("real"):+.3f}  OE control {oe("control"):+.3f}  '
                          f'CE {oe("real") - oe("control"):+.3f}  ({time.time() - t1:.0f} s)', flush=True)
            del reps
    del model
    torch.cuda.empty_cache()
    peak = f', peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB' if on_gpu else ''
    print(f'step {step}: done in {time.time() - t0:.0f} s{peak}', flush=True)


def load_records(out, name, layer, steps, fp):
    records = []
    for step in steps:
        record = json.loads(record_path(out, name, layer, step).read_text())
        if record['config_fingerprint'] != fp:
            raise RuntimeError(f'{record_path(out, name, layer, step)} was written with another configuration')
        records.append(record)
    return records


def summarise_layer(task, records):
    steps = [r['step'] for r in records]

    def seeds(kind, key):
        return np.array([[p[key] for p in r['probes'] if p['kind'] == kind] for r in records])

    oe_real, oe_ctrl = seeds('real', 'oe'), seeds('control', 'oe')
    sc_real, sc_ctrl = seeds('real', 'score'), seeds('control', 'score')
    ce = oe_real.mean(1) - oe_ctrl.mean(1)
    sel = sc_real.mean(1) - sc_ctrl.mean(1)
    sigma0 = np.array([r['sigma0'] for r in records])
    curves = dict(steps=steps, oe_real=oe_real.mean(1), oe_control=oe_ctrl.mean(1), ce=ce, sigma0=sigma0,
                  score_real=sc_real.mean(1), score_control=sc_ctrl.mean(1), selectivity=sel,
                  oe_real_seeds=oe_real, oe_control_seeds=oe_ctrl, score_real_seeds=sc_real, score_control_seeds=sc_ctrl)
    curves = {key: np.asarray(v).tolist() for key, v in curves.items()}
    late = np.array(steps) >= 128
    stats = dict(k=records[0]['k'], median_sigma0=float(np.median(sigma0)),
                 median_ce_over_sigma0_from_step128=float(np.median(ce[late] / sigma0[late])) if late.any() else None,
                 ce_positive_share=float(np.mean(ce > 0)),
                 seed_sd_over_sigma0=dict(real=float(np.median(oe_real.std(1, ddof=1) / sigma0)),
                                          control=float(np.median(oe_ctrl.std(1, ddof=1) / sigma0))))
    return dict(task=task.name, title=task.title, score=task.score_name, layer=records[0]['layer'], curves=curves,
                events=events.layer_events(steps, ce, sel), stats=stats)


def compare_layer(summary, ref):
    """Largest differences from the paper's values for the same model, task and layer."""
    r = ref['layers'][str(summary['layer'])]
    c = summary['curves']
    idx = [ref['steps'].index(s) for s in c['steps']]
    paper = dict(oe_real=np.mean(r['oe_real'], 1), oe_control=np.mean(r['oe_control'], 1),
                 score_real=np.mean(r['score_real'], 1), score_control=np.mean(r['score_control'], 1))
    paper['ce'] = paper['oe_real'] - paper['oe_control']
    paper['selectivity'] = paper['score_real'] - paper['score_control']
    out = {}
    for key, v in paper.items():
        diff = np.abs(np.array(c[key]) - v[idx])
        out[key] = dict(max_abs_diff=float(diff.max()), at_step=c['steps'][int(diff.argmax())])
    out['events'] = dict(paper=r['events'], this_run={k: summary['events'][k] for k in ('t_D', 't_E', 'lag')}) \
        if len(idx) == len(ref['steps']) else None
    ok = all(v['max_abs_diff'] <= TOLERANCE['oe' if k in ('oe_real', 'oe_control', 'ce') else 'score']
             for k, v in out.items() if k != 'events')
    if out['events']:
        mine, paper = out['events']['this_run'], out['events']['paper']
        ok = ok and abs(mine['t_D'] - paper['t_D']) < 1e-6 and mine['t_E'] == paper['t_E']
    out['matches_paper'] = ok
    return out


def print_layer(summary):
    c, ev, st = summary['curves'], summary['events'], summary['stats']
    print(f"\n{summary['title']}, layer {summary['layer']}")
    print(f"{'step':>7} {'OE real':>8} {'OE ctrl':>8} {'CE':>7} {'sigma0':>7} {'real':>8} {'control':>8} {'select.':>8}"
          f"   (scores: {summary['score']})")
    for i, step in enumerate(c['steps']):
        print(f"{step:>7} {c['oe_real'][i]:>+8.3f} {c['oe_control'][i]:>+8.3f} {c['ce'][i]:>+7.3f} {c['sigma0'][i]:>7.3f} "
              f"{c['score_real'][i]:>8.4f} {c['score_control'][i]:>8.4f} {c['selectivity'][i]:>+8.4f}")
    t_e = 'none' if ev['t_E'] is None else f"{ev['t_E']:.0f}"
    lag = 'n/a' if ev['lag'] is None else f"{ev['lag']:+.2f} ({ev['relation']})"
    print(f"t^D = {ev['t_D']:.0f}   t^E = {t_e}   lag ln(t^E/t^D) = {lag}")
    if len(c['steps']) < len(CHECKPOINTS):
        print('(event times are only meaningful on the full checkpoint grid)')
    snr = st['median_ce_over_sigma0_from_step128']
    print(f"sigma0 (k = {st['k']}): median {st['median_sigma0']:.3f} bits; median CE/sigma0 from step 128 on: "
          f"{'n/a' if snr is None else f'{snr:.1f}'}; seed SD of OE / sigma0: real {st['seed_sd_over_sigma0']['real']:.2f}, "
          f"control {st['seed_sd_over_sigma0']['control']:.2f}")
    cmp = summary.get('comparison_with_paper')
    if cmp:
        diffs = ', '.join(f"{k} {v['max_abs_diff']:.4f}" for k, v in cmp.items() if k not in ('events', 'matches_paper'))
        print(f'max |difference| from the paper: {diffs}')
        if cmp['events']:
            p = cmp['events']['paper']
            print(f"paper: t^D = {p['t_D']:.0f}, t^E = {'none' if p['t_E'] is None else round(p['t_E'])}")
        print('-> matches the paper' if cmp['matches_paper'] else
              f"-> DIFFERS from the paper beyond the tolerance (OE/CE {TOLERANCE['oe']} bits, scores {TOLERANCE['score']}"
              f"{', identical event times' if cmp['events'] else ''})")


def parse_layers(text, n):
    if text == 'all':
        return list(range(n))
    layers = sorted({int(x) for x in text.split(',')})
    if layers[0] < 0 or layers[-1] >= n:
        raise ValueError(f'Layers must be in 0..{n - 1}')
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--task', required=True, help=f"comma-separated task names or 'all': {', '.join(tasks.TASKS)}")
    ap.add_argument('--layers', required=True, help="comma-separated block indices, or 'all'")
    ap.add_argument('--model', default='pythia-410m', choices=list(MODELS))
    ap.add_argument('--checkpoints', default=','.join(map(str, CHECKPOINTS)),
                    help='comma-separated training steps (default: the 19 checkpoints of the paper)')
    ap.add_argument('--out', default=str(REPO/'runs'), help='output root; results go to OUT/<model>/ (reused to resume)')
    ap.add_argument('--data-dir', default=str(REPO/'data'), help='directory filled by scripts/download_data.sh')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--agop-batch-size', type=int, default=8, help='sequences per AGOP batch (memory only)')
    ap.add_argument('--layer-chunk', type=int, default=6, help='layers extracted per pass over the data (memory only)')
    args = ap.parse_args()
    names = list(tasks.TASKS) if args.task == 'all' else args.task.split(',')
    unknown = [n for n in names if n not in tasks.TASKS]
    if unknown:
        raise ValueError(f'Unknown task(s) {unknown}; choose from {list(tasks.TASKS)}')
    layers = parse_layers(args.layers, MODELS[args.model])
    steps = [int(s) for s in args.checkpoints.split(',')]
    if steps != sorted(set(steps)) or not set(steps) <= set(CHECKPOINTS):
        raise ValueError(f'--checkpoints must be distinct, increasing and among {CHECKPOINTS}')

    data_dir = Path(args.data_dir)
    out = Path(args.out)/args.model
    out.mkdir(parents=True, exist_ok=True)
    args.fingerprints = {}
    for name in names:
        cfg = task_config(args.model, tasks.TASKS[name])
        path = out/name/'config.json'
        if path.exists() and fingerprint(json.loads(path.read_text())) != fingerprint(cfg):
            raise RuntimeError(f'{out/name} holds results of another configuration; use another --out')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=1))
        args.fingerprints[name] = fingerprint(cfg)

    todo = {step: {n: [l for l in layers if not record_path(out, n, l, step).exists()] for n in names} for step in steps}
    todo = {step: {n: ls for n, ls in t.items() if ls} for step, t in todo.items()}
    todo = {step: t for step, t in todo.items() if t}
    if todo:
        tokenizer = lm.load_tokenizer(f'EleutherAI/{args.model}')
        print(f'{args.model}: {sum(len(ls) for t in todo.values() for ls in t.values())} (task, layer, checkpoint) cells '
              f'to run at {len(todo)} checkpoint(s)', flush=True)
        cache = {}                                    # task data, loaded once; tasks with the same key share it
        for name in names:
            cache[name] = tasks.TASKS[name].load(data_dir, tokenizer)
        ngram.statistics.cache_clear()               # the n-gram counts are only needed for the targets
        corpus = data.agop_corpus(data_dir)
        for step in todo:
            run_checkpoint(step, todo[step], args, out, cache, corpus, tokenizer)

    verdicts = {}
    for name in names:
        task = tasks.TASKS[name]
        ref_path = REPO/'reference'/args.model/f'{name}.json'
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
        per_layer = {}
        for layer in layers:
            summary = summarise_layer(task, load_records(out, name, layer, steps, args.fingerprints[name]))
            if ref:
                summary['comparison_with_paper'] = compare_layer(summary, ref)
                verdicts[(name, layer)] = summary['comparison_with_paper']['matches_paper']
            path = out/name/f'layer{layer}'
            (path/'summary.json').write_text(json.dumps(summary, indent=1))
            plot_layer(summary, args.model, path/'curves')
            print_layer(summary)
            per_layer[layer] = summary['events']
        if len(layers) == MODELS[args.model] and steps == CHECKPOINTS:
            te = events.task_events(list(per_layer.values()))
            result = dict(task=name, title=task.title, task_events=te, per_layer=per_layer,
                          paper=ref['task_events'] if ref else None)
            (out/name/'task_events.json').write_text(json.dumps(result, indent=1))
            p = ref['task_events'] if ref else None
            print(f"\n{task.title}, all {len(layers)} layers: t^D = {te['t_D']:.0f}, t^E = {te['t_E']:.0f}, "
                  f"lag {te['lag']:+.2f} ({te['relation']})"
                  + (f"; paper: t^D = {p['t_D']:.0f}, t^E = {p['t_E']:.0f}, lag {p['lag']:+.2f}" if p else ''))
    done = {n: json.loads((out/n/'task_events.json').read_text()) for n in tasks.TASKS if (out/n/'task_events.json').exists()}
    if done:
        plot_event_times(done, args.model, out/'event_times')
        print(f'\nWrote {out}/event_times.png ({len(done)} task(s) with all layers)')
    if verdicts:
        bad = [f'{n} layer {l}' for (n, l), ok in verdicts.items() if not ok]
        print(f"\n{'PASS' if not bad else 'FAIL'}: {len(verdicts) - len(bad)} of {len(verdicts)} (task, layer) results "
              f"match the paper{'' if not bad else '; differing: ' + ', '.join(bad)}")
        if bad:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
