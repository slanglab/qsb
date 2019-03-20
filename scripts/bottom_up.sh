### This should run all results for paper


echo "f1,slor mu, slor std,method" > bottom_up_clean/results.csv

TRAIN=preproc/mini.training.paths
VALIDATION=preproc/mini.validation.paths

# get slor and f1 for random
python bottom_up_clean/do_f1_and_slor.py -training_paths $TRAIN -validation_paths $VALIDATION -random

# get slor and f1 for learned
python bottom_up_clean/do_f1_and_slor.py -training_paths $TRAIN -validation_paths $VALIDATION
 
python bottom_up_clean/timer.py -path_to_set_to_evaluate $VALIDATION -N 10000
py bottom_up_clean/ilp_wrapper.py -do_jsonl preproc/validation.jsonl
