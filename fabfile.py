'''
Using this file requires version 1 of the python library fabric. 
- http://www.fabfile.org/

python3

'''
from __future__ import with_statement
from fabric import Connection
import os

local = Connection("localhost").local


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
        local("python bottom_up_clean/oracle_paths.py") # make oracle paths
        local("./scripts/proc_test.sh")  # finally, process the test set
        print("[*] preprocessed done")
    else:
        print("[*] Can't find data. Do you need to run download? Try $fab download")


def train_ilp():
    local("python ilp2013/fillipova_altun_learning.py")


def f1_and_slor():
    local("./scripts/bottom_up.sh")


def do_test():
    local("./scripts/test_results.sh")