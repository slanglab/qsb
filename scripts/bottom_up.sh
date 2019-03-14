### This should run all results for paper

rm -rf bottom_up_clean/results.csv

python bottom_up_clean/do_f1_and_slor.py -training_paths preproc/training.paths -validation_paths preproc/validation.paths
 
py bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/validation.paths -N 100
