'''deleted from fabfile'''


def computational_experiments_1():
    local("python comp_experiments_part_1/p_deletion_oracle_path.py") 
    local("Rscript scripts/oracle_acceptability_experiment.R")