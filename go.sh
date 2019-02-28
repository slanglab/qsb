#!/usr/bin/env bash
source activate lg 
export PYTHONPATH=.
t="/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/"$(date +%N)
allennlp train $1 -s $t --include-package nn 
