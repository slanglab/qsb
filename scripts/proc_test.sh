# this processes the test set. it is done on its own

ls sentence-compression/data/comp-data.eval.json.gz | parallel 'gunzip {}'

python preproc/breaker.py sentence-compression/data/comp-data.eval.json

python preproc/proc_filipova.py sentence-compression/data/comp-data.eval.jsonl

python preproc/little_test_bridge_script.py
