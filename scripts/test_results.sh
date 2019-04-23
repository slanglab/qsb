## get the results for the test set
export PYTHONPATH=.:allennlp2/allennlpallennlpasalibraryexample/
set -e
rm -rf bottom_up_clean/timing/*

echo "" > bottom_up_clean/stat_sig.csv
echo "" > bottom_up_clean/results.csv
echo "" > bottom_up_clean/timer.csv
echo "" > bottom_up_clean/all_times.csv


# get f1 and slor for additive (feature based)
python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths

# get F1 and slor for additive (random)
python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths -random

# get F1 and slor for additive (neural network)
python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths -nn

# get F1 and slor for additive (ablated)
python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths -only_locals


# get test set f1 and slor for vanilla ILP
python bottom_up_clean/ilp_wrapper.py -do_jsonl preproc/test.jsonl -ilp_snapshot 5

# get test set f1 for the vanilla ILP
python bottom_up_clean/ilp_wrapper.py -test_set_score -ilp_snapshot 5

./scripts/test_timing_results.sh

# do significance tests
python code/significance_testing.py -file1 bottom_up_clean/timing/additive_full.jsonl -file2 bottom_up_clean/timing/ilp.jsonl -metric time
python code/significance_testing.py -file1 bottom_up_clean/make_decision_lr_results.jsonl -file2 bottom_up_clean/ilp_results.jsonl -metric slor
python code/significance_testing.py -file1 bottom_up_clean/make_decision_lr_results.jsonl -file2 bottom_up_clean/ilp_results.jsonl -metric f1

# compression rate results
python bottom_up_clean/cr_results.py
