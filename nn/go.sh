#!/usr/bin/env bash
source activate lg 
export PYTHONPATH=.
t="/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/"$(date +%N)
allennlp train $1 -s $t --include-package nn 
allennlp predict $t/model.tar.gz tests/fixtures/sample.jsonl --include-package nn --predictor paper-classifier --output-file $t/preds.jsonl
