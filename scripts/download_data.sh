#!/usr/bin/env bash
# Download the data used by run_pos_layer.py and check that it is the version used in the paper.
#   bash scripts/download_data.sh [DATA_DIR]        (default: ./data)
#
#   DATA_DIR/ud_ewt/en_ewt-ud-{train,test}.conllu     UD English EWT v2.17 (CC BY-SA 4.0)
#   DATA_DIR/wikitext-103-v1/*.parquet                 WikiText-103, Salesforce/wikitext (CC BY-SA 3.0 / GFDL)
set -euo pipefail

DATA="${1:-data}"
mkdir -p "$DATA/ud_ewt" "$DATA/wikitext-103-v1"

EWT=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.17
for split in train test; do
  curl -fsSL -o "$DATA/ud_ewt/en_ewt-ud-$split.conllu" "$EWT/en_ewt-ud-$split.conllu"
done

WIKITEXT=https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-v1
for name in test-00000-of-00001 train-00000-of-00002 train-00001-of-00002 validation-00000-of-00001; do
  curl -fsSL -o "$DATA/wikitext-103-v1/$name.parquet" "$WIKITEXT/$name.parquet"
done

python - "$DATA" <<'EOF'
import sys
from pathlib import Path
from probe_agop import data

root = Path(sys.argv[1])
expected = {'train': '6fe5692f2e02198262483daba0626efe346dd19f0ce6c3ee7728b08af9b90a9c',
            'test': 'b837531faddd54f5c89cf5259cfe5306b075556a9ff19b2271393df29c6b02f8'}
for split, digest in expected.items():
    if data.sha256(root/'ud_ewt'/f'en_ewt-ud-{split}.conllu') != digest:
        sys.exit(f'en_ewt-ud-{split}.conllu is not the UD English EWT v2.17 file used in the paper')
data.load_pos_splits(root/'ud_ewt')
data.agop_corpus(root/'wikitext-103-v1')
print(f'Data in {root} matches the paper (UD EWT v2.17; AGOP corpus fingerprint {data.AGOP_CORPUS_FINGERPRINT}).')
EOF
