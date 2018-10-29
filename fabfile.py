'''
Using this file requires version 1 of the python library fabric. 
- http://www.fabfile.org/

This will activate a venv that has fabric installed
'''
from __future__ import with_statement
from fabric.api import env
from fabric.api import local, run
from code.log import logger
import os


def download():
    '''
    Download the data from Google. Will redownload if already found on machine
    '''
    local("rm -rf sentence-compression")
    local("git clone https://github.com/google-research-datasets/sentence-compression.git")
    local("rm -rf sentence-compression/*.git && mkdir -p sentence-compression/data/")


def lstm_preproc():
    local("python preproc/lstm_preproc.py")
    local("./scripts/send_to_gpu.sh")

def computational_experiments_1():
    local("python comp_experiments_part_1/p_deletion_oracle_path.py") 
    local("Rscript scripts/oracle_acceptability_experiment.R")

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
        local("python preproc/split_validation_and_test.py")
        print "[*] preprocessed done"
    else:
        print "[*] Can't find data. Do you need to run download? Try $fab download" 
