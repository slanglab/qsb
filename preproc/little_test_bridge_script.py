# coding: utf-8

# this is just a preprocessing script that performs some of the preproc used on the training/val set for compatability 

import json
from preproc.split_validation_and_training import load_dataset
from preproc.lstm_preproc import just_save_sentences

dt = load_dataset("sentence-compression/data/comp-data.eval.source")
with open("preproc/test.jsonl", "w") as of:
    dt = [json.dumps(_) for _ in dt]
    of.write("\n".join(dt))


data = [d for d in data if len(d["q"]) > 0]

just_save_sentences('sentence-compression/data/test_set.jsonl',
                    range(1, 1000000),
                    corpus="preproc/test.jsonl",
                    cap=1000000)