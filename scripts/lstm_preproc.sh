## get the results for the test set
export PYTHONPATH=.:allennlp2/allennlpallennlpasalibraryexample/

python bottom_up_clean/lstm_proc.py -fn preproc/training.paths
python bottom_up_clean/lstm_proc.py -fn preproc/validation.paths
python bottom_up_clean/lstm_proc.py -fn preproc/test.paths

scp preproc/*lstm* ahandler@gypsum.cs.umass.edu:/home/ahandler/qsr/allennlp2/allennlpallennlpasalibraryexample/experiments/
