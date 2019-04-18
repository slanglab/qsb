# just get the compression rates for the appendix 

import json
import numpy as np

lr = "bottom_up_clean/make_decision_lr_results.jsonl"
nn = "bottom_up_clean/make_decision_nn_results.jsonl"
random = "bottom_up_clean/make_decision_random_results.jsonl"
ablated = "bottom_up_clean/only_locals_results.jsonl" 
ilp = "bottom_up_clean/ilp_results.jsonl" 


def get_cr(i):
    s = " ".join([o["word"] for o in i["tokens"]])
    c = " ".join([o["word"] for o in i["tokens"] if o["index"] in i["compression_indexes"]])
    return len(c)/len(s)

test = np.mean([get_cr(json.loads(i)["sentence"]) for i in open("preproc/test.paths")])
train = np.mean([get_cr(json.loads(i)["sentence"]) for i in open("preproc/training.paths")])


lr = np.mean([json.loads(i)["cr"] for i in open(lr)])
nn = np.mean([json.loads(i)["cr"] for i in open(nn)])
random = np.mean([json.loads(i)["cr"] for i in open(random)])
ablated = np.mean([json.loads(i)["cr"] for i in open(ablated)])
ilp = np.mean([json.loads(i)["cr"] for i in open(ilp)])

print("ablated, {}".format(ablated))
print("ilp, {}".format(ilp))
print("nn, {}".format(nn))
print("random, {}".format(random))
print("lr, {}".format(lr))
print("test {}".format(test))
print("train {}".format(train))
