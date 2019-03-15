# coding: utf-8

# this is just a preprocessing script that performs some of the preproc used on the training/val set for compatability. import the actual functions used in previous to makes sure the preproc is consistent 

import json
import pickle
from preproc.split_validation_and_training import load_dataset
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform

dt = load_dataset("sentence-compression/data/comp-data.eval.source")

dt = [d for d in dt if len(d["q"]) > 0]

with open("preproc/test.jsonl", "w") as of:
    dt = [json.dumps(_) for _ in dt]
    of.write("\n".join(dt))

with open("preproc/training.ilp", "wb") as of:
    dt = [filippova_tree_transform(json.loads(i)) for i in dt]
    pickle.dump(dt, of)
