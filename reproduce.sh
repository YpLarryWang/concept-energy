#!/usr/bin/env bash
# Reproduce the POS result of the paper for one layer of Pythia-410M (run inside the environment of requirements.txt).
#   bash reproduce.sh check      environment test, ~5 min on an A100: GPU, versions, data, and 2 checkpoints
#                                (steps 512 and 143000, layer 8) compared with the paper -> PASS or FAIL
#   bash reproduce.sh [LAYER]    the full result: all 19 checkpoints of one layer (default 8), ~30 min on an A100
# Data is downloaded on the first call (scripts/download_data.sh); Pythia checkpoints come from Hugging Face on first use.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-8}"

python - <<'EOF'
import sys
import torch
import transformers
print(f'Python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}')
if not torch.cuda.is_available():
    sys.exit('No CUDA GPU is visible to torch; run_pos_layer.py needs an NVIDIA GPU.')
gpu = torch.cuda.get_device_properties(0)
print(f'GPU: {gpu.name}, {gpu.total_memory / 2**30:.0f} GB')
for name, have, tested in [('torch', torch.__version__.split('+')[0], '2.7.1'),
                           ('transformers', transformers.__version__, '4.57.3')]:
    if have != tested:
        print(f'Note: {name} {have} is not the tested version {tested}; expect small numerical differences.')
EOF

if [ ! -f data/ud_ewt/en_ewt-ud-train.conllu ] || [ "$(ls data/wikitext-103-v1/*.parquet 2>/dev/null | wc -l)" -ne 4 ]; then
  bash scripts/download_data.sh data
fi

if [ "$MODE" = check ]; then
  python run_pos_layer.py --layer 8 --checkpoints 512,143000 --out runs/check
  python - <<'EOF'
import json
import sys
diff = json.load(open('runs/check/summary.json'))['comparison_with_paper']
tolerance = dict(oe_real=0.01, oe_control=0.01, ce=0.01, acc_real=0.002, acc_control=0.002, selectivity=0.002)
over = {k: round(diff[k]['max_abs_diff'], 4) for k, t in tolerance.items() if diff[k]['max_abs_diff'] > t}
if over:
    sys.exit(f'\nFAIL: differences from the paper {over} exceed the tolerances {tolerance}')
print(f'\nPASS: OE, CE (tolerance 0.01 bits) and accuracies (tolerance 0.002) match the paper at steps 512 and 143000.')
EOF
else
  python run_pos_layer.py --layer "$MODE" --out "runs/pythia-410m_pos_layer$MODE"
fi
