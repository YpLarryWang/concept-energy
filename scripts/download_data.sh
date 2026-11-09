#!/usr/bin/env bash
# Download the data of all tasks and check that every file is the version used in the paper.
#   bash scripts/download_data.sh [DATA_DIR]        (default: ./data)
#
#   DATA_DIR/ud_ewt/en_ewt-ud-{train,test}.conllu        UD English EWT v2.17 (CC BY-SA 4.0)
#   DATA_DIR/uner_ewt/en_ewt-ud-{train,test}.iob2        UNER English EWT, commit 8ed072de (CC BY-SA 4.0)
#   DATA_DIR/streusle/streusle.ud_{train,test}.conllu    STREUSLE 5.0 (CC BY-SA 4.0)
#   DATA_DIR/wikitext-103-v1/*.parquet                    WikiText-103, Salesforce/wikitext (CC BY-SA 3.0 / GFDL)
set -euo pipefail

DATA="${1:-data}"
mkdir -p "$DATA/ud_ewt" "$DATA/uner_ewt" "$DATA/streusle" "$DATA/wikitext-103-v1"

EWT=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.17
UNER=https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/8ed072de2fd24022cc62458997fe96a8fe191ea4
STREUSLE=https://raw.githubusercontent.com/nert-nlp/streusle/8ba61fe4f216e7967500a862554a4fff79d25f5d
for split in train test; do
  curl -fsSL -o "$DATA/ud_ewt/en_ewt-ud-$split.conllu" "$EWT/en_ewt-ud-$split.conllu"
  curl -fsSL -o "$DATA/uner_ewt/en_ewt-ud-$split.iob2" "$UNER/en_ewt-ud-$split.iob2"
  curl -fsSL -o "$DATA/streusle/streusle.ud_$split.conllu" "$STREUSLE/$split/streusle.ud_$split.conllu"
done

WIKITEXT=https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-v1
for name in test-00000-of-00001 train-00000-of-00002 train-00001-of-00002 validation-00000-of-00001; do
  curl -fsSL -o "$DATA/wikitext-103-v1/$name.parquet" "$WIKITEXT/$name.parquet"
done

python -m probe_agop.data "$DATA"
