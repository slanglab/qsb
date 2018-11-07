#!/usr/bin/env bash
source activate allennlp

rm comp_experiments_f1/output/*
python comp_experiments_f1/run_sentence.py -config comp_experiments_f1/experiments/prune_only_nn.json
python comp_experiments_f1/run_sentence.py -config comp_experiments_f1/experiments/ilp.json
python comp_experiments_f1/consolidator.py
