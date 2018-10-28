#!/usr/bin/env bash
source activate allennlp
export PYTHONPATH=.
t="experiments/compressor/"$(date +%s)
allennlp train $1 -s $t --include-package nn 
allennlp predict $t/model.tar.gz tests/fixtures/s2_papers.jsonl --include-package nn --predictor paper-classifier --output-file $t/preds.jsonl
