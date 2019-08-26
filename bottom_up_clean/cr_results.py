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

print("ablated, {}".format(round(ablated,3)))
print("ilp, {}".format(round(ilp,3)))
print("nn, {}".format(round(nn,3)))
print("random, {}".format(round(random,3)))
print("lr, {}".format(round(lr,3)))
print("test {}".format(round(test,3)))
print("train {}".format(round(train,3)))
