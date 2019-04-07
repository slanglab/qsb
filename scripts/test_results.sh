## get the results for the test set
set -e
rm -rf bottom_up_clean/timing/*

echo "" > bottom_up_clean/stat_sig.csv
echo "" > bottom_up_clean/results.csv
echo "" > bottom_up_clean/timer.csv

# get f1 and slor for additive + ablated
python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths

python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths -only_locals

# get test set f1 and slor for vanilla ILP
python bottom_up_clean/ilp_wrapper.py -do_jsonl preproc/test.jsonl -ilp_snapshot 5

# get test set f1 for the vanilla ILP
python bottom_up_clean/ilp_wrapper.py -test_set_score -ilp_snapshot 5

# run timing experiments
python bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/test.paths -N 100000 -ilp_snapshot 5

# do significance tests on time
python code/significance_testing.py -file1 bottom_up_clean/timing/random.jsonl -file2 bottom_up_clean/timing/ilp.jsonl -metric time
python code/significance_testing.py -file1 bottom_up_clean/timing/additive_full.jsonl -file2 bottom_up_clean/timing/ilp.jsonl -metric time
python code/significance_testing.py -file1 bottom_up_clean/timing/additive_full.jsonl -file2 bottom_up_clean/timing/ilp.jsonl -metric time
python code/significance_testing.py -file1 bottom_up_clean/timing/additive_ablated.jsonl -file2 bottom_up_clean/timing/ilp.jsonl -metric time


# do results on slor/f1
python code/significance_testing.py -file1 bottom_up_clean/additive_results.jsonl -file2 bottom_up_clean/ilp_results.jsonl -metric slor
python code/significance_testing.py -file bottom_up_clean/additive_results.jsonl -file2 
