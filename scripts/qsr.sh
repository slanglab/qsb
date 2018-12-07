#!/usr/bin/env bash
source activate allennlp

rm comp_experiments_f1/output/*

# assess the convergence of ILP w/ validation set
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp1.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp2.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp3.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp4.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp5.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp6.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp7.json
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp8.json

# validation performance of transition-based compression, constrained
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/nn-prune-greedy.json

# test performance of transition-based compression, constrained
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/nn-prune-greedy.json -test

# test performance of constrained ILP, using best validation weights
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/ilp7.json -test

# test performance of unconstrained ILP, using best validation weights
python comp_experiments_f1/run_f1_experiment.py -config comp_experiments_f1/experiments/standard-ilp7.json -test

python comp_experiments_f1/consolidator.py
