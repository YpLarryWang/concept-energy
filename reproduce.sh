#!/usr/bin/env bash
# One command per result of the paper. Run inside the environment of requirements.txt, on a machine with an NVIDIA GPU.
#
#   bash reproduce.sh                        POS at layer 8 of Pythia-410M over the 19 checkpoints (~35 min on an A100):
#                                            environment check, data download, run, figure, the paper's values
#   bash reproduce.sh check                  quick test (~20 min): every task at layer 8, steps 512 and 143000
#   bash reproduce.sh TASK [LAYER] [MODEL]   any task, layer (default 8) and model (pythia-410m or pythia-1b)
#   bash reproduce.sh all [MODEL]            all 12 tasks x all layers x 19 checkpoints, with the task-level event times
#                                            (about a day per model on one A100)
#
# Every run prints the paper's values (reference/) next to its own. Data is downloaded on the first
# call (scripts/download_data.sh); Pythia checkpoints come from Hugging Face on first use. Results go to runs/.
set -euo pipefail
cd "$(dirname "$0")"
WHAT="${1:-pos}"

python - <<'EOF'
import sys
import torch
import transformers
print(f'Python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}')
if not torch.cuda.is_available():
    sys.exit('No CUDA GPU is visible to torch; the experiments need an NVIDIA GPU.')
gpu = torch.cuda.get_device_properties(0)
print(f'GPU: {gpu.name}, {gpu.total_memory / 2**30:.0f} GB')
for name, have, tested in [('torch', torch.__version__.split('+')[0], '2.7.1'),
                           ('transformers', transformers.__version__, '4.57.3')]:
    if have != tested:
        print(f'Note: {name} {have} is not the tested version {tested}; expect small numerical differences.')
EOF

if ! python -m probe_agop.data data > /dev/null 2>&1; then
  bash scripts/download_data.sh data
fi

case "$WHAT" in
  check)
    python run.py --task all --layers 8 --checkpoints 512,143000 --out runs/check ;;
  all)
    python run.py --task all --layers all --model "${2:-pythia-410m}" --out runs ;;
  *)
    python run.py --task "$WHAT" --layers "${2:-8}" --model "${3:-pythia-410m}" --out runs ;;
esac
