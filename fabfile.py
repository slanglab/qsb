'''
Using this file requires version 1 of the python library fabric. 
- http://www.fabfile.org/

python3

'''
from __future__ import with_statement
from fabric import Connection
from fabric import task
from code.log import logger
import os

local = Connection("localhost").local


def bottom_up():
    local("python bottom_up_clean/oracle_paths.py && py bottom_up_clean/driver4.py -training_paths preproc/training.paths -validation_paths preproc/validation.paths")


def download():
    '''
    Download the data from Google. Will redownload if already found on machine
    '''
    local("rm -rf sentence-compression")
    local("git clone https://github.com/google-research-datasets/sentence-compression.git")
    local("rm -rf sentence-compression/*.git && mkdir -p sentence-compression/data/")


def preproc():
    '''
    Run preprocessing for fillipova corpus

    uses ~/rsumdata/stanford-corenlp-full-2017-06-09/run_server.sh for corenlp
    '''
    if (os.path.exists("sentence-compression")):
        local("ls sentence-compression/data/*train*gz | parallel 'gunzip {}'")
        local("ls sentence-compression/data/*json | parallel 'python preproc/breaker.py {}'")
        local("ls sentence-compression/data/*json | parallel rm")
        # you need to run server in stanford-corenlp-full-2017-06-09/run_server.sh
        local("ls sentence-compression/data/*jsonl | parallel -j 5 --eta 'python preproc/proc_filipova.py {}'") # do corenlp processing
        local("ls sentence-compression/data/*sent-comp.train*jsonl | parallel rm")
        local("python preproc/split_validation_and_training.py")
        local("./scripts/proc_test.sh")  # finally, process the test set
        print("[*] preprocessed done")
    else:
        print("[*] Can't find data. Do you need to run download? Try $fab download")


def lstm_preproc():
    '''preprocess the data for the lstm'''
    local("python preproc/lstm_preproc.py")
    local("./scripts/send_to_gpu.sh")
    # You also need to make a full_counts.txt file using the training set.
    # This file is used to weight the objective function in nn/models/two_way_shared_params.py
    # $ cat preproc/some_lstm_training_file.jsonl | jq .label | sort | uniq -c > preproc/full_counts.txt"


@task
def qsr(cx):
    '''run the (q,s,r) F1 experiments'''
    local("./scripts/qsr.sh")


@task
def complexity(cx):
    '''run the complexity experiments'''
    local("python comp_experiments_complexity/preprocess_complexity_plots.py comp_experiments_f1/output/full-worst-case-worst-case-compressor-test")
    local("python comp_experiments_complexity/preprocess_complexity_plots.py comp_experiments_f1/output/full-556251071-nn-prune-greedy-test")
    local("Rscript scripts/empirical_ops.R")
