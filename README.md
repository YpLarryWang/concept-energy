# Geometrically Disentangling Concept Learning from the Language Modeling Loss

Code for the NeurIPS 2026 paper by Yupei Wang, Neil Mallinar, Mikhail Belkin and Alex Warstadt (UC San Diego).

A linear probe tells us that a concept can be read out of a layer. This code asks a different question: does the language-modeling loss itself care about the probe's directions? At each pretraining checkpoint we compare the probe's weight subspace with the average gradient outer product (AGOP) of the next-token loss at the same layer, and track how the two line up over training.

## One command

You need an NVIDIA GPU with CUDA.

```bash
git clone https://github.com/YpLarryWang/concept-energy.git && cd concept-energy
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash reproduce.sh
```

`bash reproduce.sh` checks the environment (GPU, package versions), downloads the data on the first call, and reproduces the POS result at layer 8 of Pythia-410M over the paper's 19 checkpoints (step 1 to 143k, about 35 minutes on an A100). It runs these steps:

1. at every checkpoint, compute the AGOP $G$ of the next-token loss with respect to the layer's output, on up to 8 sampled positions in each of 1,000 WikiText-103 lines;
2. at every checkpoint, extract the layer's word representations on UD English EWT and train 3 real probes (16 UPOS classes) and 3 control probes (each word type gets one fixed random label, drawn with the class shares of the real labels);
3. measure each probe's Optimization Engagement (OE) with $G$, and the null noise scale $\sigma_0$ of OE;
4. compute the Concept Energy (CE) and the decodability, which is the real probes' test accuracy minus the control probes' accuracy on their own labels;
5. find the event times $t^D$ (fastest rise of decodability) and $t^E$ (first clear peak of CE), and their lag $\ln(t^E / t^D)$;
6. plot OE, CE and decodability over training, and print the paper's values for the same layer next to the run's (the paper's values are in `reference/`).

With $\ell_s$ the next-token cross-entropy at a sampled position $s$ and $h_s$ the layer's output there,

$$G = \frac{1}{N} \sum_{s=1}^{N} g_s g_s^\top, \qquad g_s = \frac{\partial \ell_s}{\partial h_s}.$$

For a probe with $K$ classes, let $U \in \mathbb{R}^{d \times k}$ hold the top $k = K - 1$ right singular vectors of its class-centered weights, where $d$ is the hidden size. Then

$$\mathrm{OE} = \log_2 \frac{\operatorname{tr}(U^\top G U) / \operatorname{tr}(G)}{k / d}, \qquad \mathrm{CE} = \overline{\mathrm{OE}}_{\text{real}} - \overline{\mathrm{OE}}_{\text{control}},$$

where the bars average over the three seeds. $\mathrm{OE} = 0$ means that the probe's subspace captures the same share of $\operatorname{tr}(G)$ as a uniformly random subspace of the same dimension does in expectation, $k/d$; each additional unit doubles that share.

![OE, CE and decodability of POS probes at layer 8 of Pythia-410M](assets/pythia-410m_pos_layer8.png)

## Every task, layer and model

The same command runs any of the paper's 12 tasks:

```bash
bash reproduce.sh TASK [LAYER] [MODEL]     # e.g. bash reproduce.sh trigram_pmi 12 pythia-1b
bash reproduce.sh check                    # quick test: every task at layer 8, steps 512 and 143000 (~20 min on an A100)
bash reproduce.sh all [MODEL]              # all tasks x all layers x 19 checkpoints, plus the event-time figure
```

| Task | In the paper | Data | Probe reads | Labels or targets | Score |
|---|---|---|---|---|---|
| `pos` | `POS` | UD English EWT | each non-punctuation word | 16 UPOS tags | test accuracy |
| `dep_relation` | `DependencyRelation` | UD English EWT | each non-punctuation word | 30 relations to the head (subtype stripped) | test accuracy |
| `dep_length` | `DependencyLength` | UD English EWT | each non-punctuation word | ROOT, or head left/right at distance 1, 2, 3–5, 6+ | test accuracy |
| `dep_depth` | `DependencyDepth` | UD English EWT | each non-punctuation word | arcs up to the root: 0–10, 11+ | test accuracy |
| `ner` | `NER` | UNER English EWT | entity tokens | PER, ORG, LOC | test accuracy |
| `noun_supersense` | `SupersenseNoun` | STREUSLE 5.0 | single-word noun expressions | 12 supersenses | test accuracy |
| `verb_supersense` | `SupersenseVerb` | STREUSLE 5.0 | single-word verb expressions | 9 supersenses | test accuracy |
| `unigram_lp` | `UnigramLogProb` | WikiText-103 | token $w_t$ | $\log p(w_t)$ | held-out $R^2$ |
| `bigram_lp` | `BigramLogProb` | WikiText-103 | token $w_t$ | $\log p(w_t \mid w_{t-1})$ | held-out $R^2$ |
| `bigram_pmi` | `BigramPMI` | WikiText-103 | token $w_t$ | $\log p(w_t \mid w_{t-1}) - \log p(w_t)$ | held-out $R^2$ |
| `trigram_lp` | `TrigramLogProb` | WikiText-103 | token $w_t$ | $\log p(w_t \mid w_{t-2} w_{t-1})$ | held-out $R^2$ |
| `trigram_pmi` | `TrigramPMI` | WikiText-103 | token $w_t$ | $\log p(w_t \mid w_{t-2} w_{t-1}) - \log p(w_t \mid w_{t-1})$ | held-out $R^2$ |

- **Classification tasks.** Classes with at least 100 training and 10 test words are kept, and every word is read at its last sub-word token, with its whole sentence as context. The control task gives each word type one fixed random label, drawn with the class shares of the real training labels.
- **N-gram regression tasks.** The targets are counted on WikiText-103 with the Pythia tokenizer, and the probe reads the last token $w_t$ of each n-gram in its line. The training and test sets are split by n-gram type, so no test n-gram occurs in training. The probe is a linear regression, so $k = 1$ and $U$ is its normalized weight vector. The control task gives each n-gram type a target drawn at random from the token-level targets of the same set (training or test).

Details are in the docstrings of `probe_agop/tasks.py` and `probe_agop/ngram.py`.

**Layers** are numbered from 0, and layer $L$ is the output of transformer block $L$: 0–23 for Pythia-410M and 0–15 for Pythia-1B. The paper's figures number layers from 1, so layer 8 here is layer 9 there.

`reproduce.sh` calls `run.py`, which takes any combination directly:

```bash
python run.py --task pos,ner,unigram_lp --layers 0,8,23 --model pythia-410m
```

| Argument | Default | Meaning |
|---|---|---|
| `--task` | (required) | comma-separated task names, or `all` |
| `--layers` | (required) | comma-separated layers (0–23 for Pythia-410M, 0–15 for Pythia-1B), or `all` |
| `--model` | `pythia-410m` | `pythia-410m` or `pythia-1b` |
| `--checkpoints` | the paper's 19 steps | comma-separated training steps |
| `--out` | `runs` | output root; results go to `OUT/<model>/` and a repeated command resumes |
| `--data-dir` | `data` | where `scripts/download_data.sh` put the data |
| `--agop-batch-size`, `--layer-chunk` | 8, 6 | memory only: sequences per AGOP batch, layers extracted per pass over the data |

**Time and memory.** On an A100 (40 GB), the AGOP of all layers at one checkpoint takes under a minute, and one probe takes about 6 s on the large UD tasks (177k words) and less on the others. One task at one layer takes 10–40 minutes, and `all` takes roughly a day per model. Several `run.py` processes with different `--task` or `--layers` can share one GPU and one `--out`. Pythia-410M uses at most about 5 GB of GPU memory. The first n-gram run counts n-grams on WikiText-103 once (about 5 minutes and 10 GB of RAM) and caches the targets in `data/ngram/`.

## Output

```
runs/pythia-410m/
  agop/step*/layer*.npy           the AGOP matrices (float64, d x d), shared by all tasks
  pos/config.json                 the full configuration; a rerun with another configuration into the same folder stops
  pos/layer8/step*.json           per checkpoint: OE and score of each probe, sigma_0, participation ratio of G
  pos/layer8/summary.json         all curves (per seed and averaged), events, sigma_0 statistics, the paper's values
  pos/layer8/curves.png, .pdf     OE (real, control), CE with +-sigma_0, decodability; t^E and t^D marked
  pos/task_events.json            with all layers: the task-level event times (the paper's event-time figure)
  event_times.png, .pdf           with all layers of one or more tasks: task-level t^D against t^E
```

The run also prints a table of all checkpoints and, for each task and layer, the paper's values next to the run's.

## Precomputed probes and AGOP matrices

All probes and AGOP matrices behind the paper's numbers are on Hugging Face, in the organization [disentangle-concept-learning](https://huggingface.co/disentangle-concept-learning): one dataset with the AGOP matrices of both models at every layer and checkpoint (`agop`), and one dataset of probes per task (`cls-*` and `reg-*`; 54,720 probes in all, each with its OE and test score). Their pages show how to compute OE from these files without running a language model.

## How the code is organized

| File | Content |
|---|---|
| `reproduce.sh` | environment check, data download and the runs above |
| `run.py` | the experiment: AGOP, probes, OE, CE, decodability, $\sigma_0$, events, figures |
| `probe_agop/data.py` | reading UD English EWT, UNER English EWT, STREUSLE and WikiText-103 (with checks of every file) |
| `probe_agop/tasks.py` | the 12 tasks: labels, classes, rows, control labels, probe fitting |
| `probe_agop/ngram.py` | n-gram counts, targets, the training/test split and the control targets |
| `probe_agop/controls.py` | control labels of the classification tasks |
| `probe_agop/model.py` | loading checkpoints; reading block outputs with forward hooks; word and token representations |
| `probe_agop/agop.py` | the AGOP of the next-token loss at sampled positions, for several layers at once |
| `probe_agop/probe.py` | input centering and the probe training recipe (Adam, 10 epochs) |
| `probe_agop/energy.py` | OE, the participation ratio of $G$, and $\sigma_0$ |
| `probe_agop/events.py` | $t^D$, $t^E$, the lag, and the layer-weighted task-level event times |
| `probe_agop/plot.py` | the figures |
| `reference/<model>/<task>.json` | the paper's OE and scores per layer, checkpoint and seed, and its event times |
| `tests/test_metrics.py` | CPU checks of the metrics, including the paper's event times of every task and model |

The tests need no GPU and no data download:

```bash
pytest tests
```

## Data and licenses

| Data | Used for | Version | License |
|---|---|---|---|
| [Pythia](https://github.com/EleutherAI/pythia) checkpoints | all experiments | `step1` … `step143000` on Hugging Face | Apache-2.0 |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | AGOP corpus; n-gram tasks | `wikitext-103-v1` | CC BY-SA 3.0, GFDL |
| [UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT) | POS and dependency tasks | v2.17 | CC BY-SA 4.0 |
| [UNER English EWT](https://github.com/UniversalNER/UNER_English-EWT) | NER | commit `8ed072de` | CC BY-SA 4.0 |
| [STREUSLE](https://github.com/nert-nlp/streusle) | noun and verb supersenses | 5.0 (commit `8ba61fe4`) | CC BY-SA 4.0 |

`scripts/download_data.sh` fetches the exact files used in the paper and checks their SHA-256, and `transformers` downloads the Pythia checkpoints on first use. The paper's behavioral analyses (subject–verb agreement with BLiMP, output KL) and robustness checks are not part of this repository.

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

Code: Apache-2.0 (see `LICENSE`).
