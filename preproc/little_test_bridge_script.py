# coding: utf-8

# this is just a preprocessing script that performs some of the preproc used on the training/val set for compatability 

import json
from preproc.split_validation_and_training import load_dataset
dt = load_dataset("sentence-compression/data/comp-data.eval.source")
with open("preproc/test.jsonl", "w") as of:
    dt = [json.dumps(_) for _ in dt]
    of.write("\n".join(dt))
