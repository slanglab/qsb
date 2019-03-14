'''deleted from fabfile'''

def lstm_preproc():
    '''preprocess the data for the lstm'''
    local("python preproc/lstm_preproc.py")
    local("./scripts/send_to_gpu.sh")
    # You also need to make a full_counts.txt file using the training set.
    # This file is used to weight the objective function in nn/models/two_way_shared_params.py
    # $ cat preproc/some_lstm_training_file.jsonl | jq .label | sort | uniq -c > preproc/full_counts.txt"


@task
def bottom(cx):
    local("python bottom_up_clean/oracle_paths.py && py bottom_up_clean/driver4.py -training_paths preproc/training.paths -validation_paths preproc/validation.paths")

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


def computational_experiments_1():
    local("python comp_experiments_part_1/p_deletion_oracle_path.py") 
    local("Rscript scripts/oracle_acceptability_experiment.R")
