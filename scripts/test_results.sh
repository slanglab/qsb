## get the results for the test set
echo "" > bottom_up_clean/stat_sig.csv
python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -skip_training -validation_paths preproc/test.paths
python bottom_up_clean/ilp_wrapper.py -do_jsonl preproc/test.jsonl -ilp_snapshot 5
py bottom_up_clean/ilp_wrapper.py -test_set_score -ilp_snapshot 5
python code/significance_testing.py
python bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/test.paths -N 5 -ilp_snapshot 5
