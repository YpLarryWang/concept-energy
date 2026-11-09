# Geometrically Disentangling Concept Learning from the Language Modeling Loss

Code for the NeurIPS 2026 paper by Yupei Wang, Neil Mallinar, Mikhail Belkin and Alex Warstadt (UC San Diego).

A linear probe tells us that a concept can be read out of a layer. This code asks a different question: does the
language-modelling loss itself care about the probe's directions? At each pretraining checkpoint we compare the probe's
weight subspace with the average gradient outer product (AGOP) of the next-token loss at the same layer, and track how
the two line up over training.

## One command, one result

```bash
git clone https://github.com/YpLarryWang/concept-energy.git && cd concept-energy
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash reproduce.sh check     # environment test (~5 min): GPU, data, 2 checkpoints compared with the paper -> PASS/FAIL
bash reproduce.sh           # the result: POS probes at layer 8 of Pythia-410M over 19 checkpoints (~30 min)
```

`bash reproduce.sh` runs `python run_pos_layer.py --layer 8 --out runs/pythia-410m_pos_layer8` (`bash reproduce.sh 12`
for layer 12). For one layer of Pythia-410M and each of the paper's 19 checkpoints (step 1 to 143k), this

1. computes the **AGOP** `G` of the next-token loss with respect to the layer's residual stream (1,000 WikiText-103 lines,
   8 sampled positions each);
2. extracts the layer's word representations on UD English EWT and trains **3 real POS probes** (16 UPOS classes) and
   **3 type-matched control probes** (each word type gets a fixed random label with the real class shares);
3. measures each probe's **Optimization Engagement**: with `U` the top `k = 15` right singular vectors of the probe's
   class-centred weights, `OE = log2( (tr(UᵀGU) / tr G) / (k / d) )` (0 = a random subspace of the same size);
4. reports **Concept Energy** `CE = OE(real) − OE(control)`, **decodability** as selectivity
   (real accuracy − control accuracy), the null noise scale **σ₀** of OE, and the event times **t^D** (fastest rise of
   decodability), **t^E** (first clear peak of CE) and their lag `ln(t^E / t^D)`;
5. draws the three curves and compares every number with the paper's values for the same layer.

Results are written per checkpoint, so an interrupted run resumes when the command is repeated. The figure of the
command above:

![OE, CE and decodability of POS probes at layer 8 of Pythia-410M](assets/pythia-410m_pos_layer8.png)

## Requirements and details

- Linux with one NVIDIA GPU. Tested on an A100 (40 GB): about 95 s per checkpoint, at most 4.6 GB of GPU memory
  allocated by PyTorch, so a GPU with 8 GB or more should suffice. Other GPUs should work but are untested; the check compares with the paper within
  tolerances (0.01 bits for OE and CE, 0.002 for accuracies) because float16 results may differ slightly across GPUs.
- Python 3.12 with the pinned versions in `requirements.txt` (tested in a fresh virtual environment). The CPU-only
  metric tests run with `python -m pytest tests`.
- Data: the first call of `reproduce.sh` runs `bash scripts/download_data.sh`, which downloads UD English EWT v2.17 and
  WikiText-103 (~320 MB) into `data/` and checks that they are the versions used in the paper. `transformers` downloads
  the Pythia checkpoints (about 0.9 GB each) into the Hugging Face cache on first use.

`run_pos_layer.py` can also be called directly:

| Argument | Default | Meaning |
|---|---|---|
| `--layer` | (required) | transformer block whose output is probed (0–23 for Pythia-410M) |
| `--out` | (required) | output directory; reused to resume |
| `--model` | `EleutherAI/pythia-410m` | any Pythia model with the same checkpoint names |
| `--checkpoints` | the paper's 19 steps | comma-separated training steps |
| `--data-dir` | `data` | where `scripts/download_data.sh` put the data |
| `--device` | `cuda` | |
| `--agop-batch-size` | `8` | sequences per AGOP batch (memory only; the result does not change) |

## Output

```
runs/pythia-410m_pos_layer8/
  curves.png, curves.pdf      OE (real, control), CE with ±σ₀, decodability; t^E and t^D marked
  summary.json                all curves (per seed and averaged), events, σ₀ statistics, comparison with the paper
  checkpoints/step*.json      per checkpoint: OE and test accuracy of each probe, σ₀, participation ratio of G
  agop/step*.npy              the AGOP matrices (float64, 1024 x 1024)
  config.json                 the full configuration; a rerun with another configuration into the same directory stops
```

The run also prints a table of all checkpoints and, for Pythia-410M, the largest difference from the paper's values
over the checkpoints. The paper's per-layer values for POS (all 24 layers, per seed) are in
`reference/pythia-410m_pos.json`. For layer 8 on an A100, all 114 test accuracies (19 checkpoints x 6 probes) are
identical to the paper's, OE and CE differ by at most 0.003 bits (about 0.06 σ₀; the AGOP is recomputed rather than
read from the paper's cache), and the event times are the same (t^D = 716, t^E = 512).

## How the code is organised

| File | Content |
|---|---|
| `reproduce.sh` | environment test (`check`) and the one-command run |
| `run_pos_layer.py` | the experiment above |
| `probe_agop/data.py` | UD EWT words and UPOS labels; the WikiText-103 AGOP corpus (with the paper's sentence filter) |
| `probe_agop/model.py` | loading checkpoints; reading a block's output with a forward hook; word representations |
| `probe_agop/agop.py` | the AGOP of the next-token loss at sampled positions |
| `probe_agop/probe.py` | input centring and the linear-probe training recipe (Adam, 10 epochs) |
| `probe_agop/controls.py` | type-matched control labels |
| `probe_agop/energy.py` | OE, the participation ratio of G, and σ₀ |
| `probe_agop/events.py` | t^D, t^E, the lag, and the layer-weighted task-level event times |
| `probe_agop/plot.py` | the figure |
| `tests/test_metrics.py` | checks of the metric definitions, including the paper's POS event times from `reference/` |

Running `run_pos_layer.py` for every layer and passing the per-layer events to `probe_agop.events.task_events` gives the
task-level point of POS in the paper's event-time figure (Pythia-410M: t^D ≈ 493, t^E ≈ 479, same time).

## Data and licenses

| Data | Used for | Version | License |
|---|---|---|---|
| [Pythia](https://github.com/EleutherAI/pythia) checkpoints | all experiments | `step1` … `step143000` on Hugging Face | Apache-2.0 |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | AGOP corpus; n-gram probes | `wikitext-103-v1` | CC BY-SA 3.0, GFDL |
| [UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT) | POS and dependency probes | v2.17 | CC BY-SA 4.0 |
| [UNER English EWT](https://github.com/UniversalNER/UNER_English-EWT) | NER probe | commit `8ed072de` | CC BY-SA 4.0 |
| [STREUSLE](https://github.com/nert-nlp/streusle) | noun and verb supersense probes | 5.0 | CC BY-SA 4.0 |
| [LGD](https://github.com/yoavg/bert-syntax) | subject–verb agreement probe | as in `bert-syntax` | see the repository |
| [BLiMP](https://github.com/alexwarstadt/blimp) | subject–verb agreement behaviour | | CC BY 4.0 |

`scripts/download_data.sh` fetches WikiText-103 and UD English EWT; `transformers` downloads the Pythia checkpoints on
first use. The other tasks of the paper (n-gram statistics, dependencies, NER, supersenses, subject–verb agreement) and
Pythia-1B are not in this repository yet.

## Citation

```bibtex
@inproceedings{wang2026geometrically,
  title     = {Geometrically Disentangling Concept Learning from the Language Modeling Loss},
  author    = {Wang, Yupei and Mallinar, Neil and Belkin, Mikhail and Warstadt, Alex},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

See `LICENSE`.
