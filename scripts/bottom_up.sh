### This should run all results for paper

python bottom_up_clean/oracle_paths.py
python bottom_up_clean/do_f1.py -training_paths preproc/training.paths -validation_paths preproc/validation.paths
# klm.query.py

 
py bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/validation.paths -N 100
