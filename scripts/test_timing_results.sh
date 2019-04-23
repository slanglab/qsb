# run timing experiments
echo "" > bottom_up_clean/timer.csv
echo "" > bottom_up_clean/all_times.csv
# get 3 seperate measurements 1 hour apart 
python bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/test.paths -N 100000 -ilp_snapshot 5
sleep 1h
python bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/test.paths -N 100000 -ilp_snapshot 5
sleep 1h
python bottom_up_clean/timer.py -path_to_set_to_evaluate preproc/test.paths -N 100000 -ilp_snapshot 5
