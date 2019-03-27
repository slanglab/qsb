### Run all results for paper

# reset the results files
echo "f1,slor mu, slor std,method" > bottom_up_clean/results.csv
echo "mu,sigma,method" > bottom_up_clean/timer.csv

TRAIN=preproc/training.paths
VALIDATION=preproc/validation.paths

# get slor and f1 only locals
python bottom_up_clean/do_f1_and_slor.py -only_locals -training_paths $TRAIN -validation_paths $VALIDATION

# get slor and f1 for learned + full
python bottom_up_clean/do_f1_and_slor.py -training_paths $TRAIN -validation_paths $VALIDATION

# get slor and f1 for random
python bottom_up_clean/do_f1_and_slor.py -training_paths $TRAIN -validation_paths $VALIDATION -random

# Run the timer results
python bottom_up_clean/timer.py -path_to_set_to_evaluate $VALIDATION -N 10000

# get the ILP F1 results
#python bottom_up_clean/ilp_wrapper.py -do_jsonl preproc/validation.jsonl
