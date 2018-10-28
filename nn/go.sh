#!/usr/bin/env bash
source activate allennlp
export PYTHONPATH=.
t="experiments/"$(date +%s)
allennlp train $1 -s $t --include-package lib
allennlp predict $t/model.tar.gz tests/fixtures/s2_papers.jsonl --include-package lib --predictor paper-classifier --output-file $t/preds.jsonl
