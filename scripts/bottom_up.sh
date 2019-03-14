python bottom_up_clean/oracle_paths.py
python bottom_up_clean/driver4.py -training_paths preproc/training.paths -validation_paths preproc/validation.paths
# klm.query.py 
python bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/validation.paths
