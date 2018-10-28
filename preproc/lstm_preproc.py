# coding: utf-8

import json

import random

from code.treeops import dfs

c = 0

PCT_TRAIN = .95

CORPUS = 'preproc/training.jsonl'

START = "OOVSTART"
END = "OOVEND"

with open(CORPUS, 'r') as inf:
    for _ in inf:
        c += 1

c = range(c)
random.shuffle(c)

split_ix = int(len(c) * PCT_TRAIN)

train = c[0:split_ix]
test = c[split_ix:]

val_ix = int((1 - PCT_TRAIN) * len(train))

val = train[0:val_ix]
train = train[val_ix:]


def save_split(fn, data):
    '''note, avoiding pulling a whole big corpus into memory so this can scale up'''
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, _ in enumerate(inf):
                    if ino in data:
                        _ = json.loads(_)
                        del _["basicDependencies"]
                        del _["len"]
                        del _["index"]
                        del _["guid"]
                        del _["enhancedDependencies"]
                        del _["enhancedPlusPlusDependencies"]
                        toks = [i["word"] for i in _["tokens"]]
                        for vertex in _["tokens"]:
                            oracle = _["oracle"][str(vertex)]
                            _['label'] = oracle
                            cut = dfs(g=_, hop_s = v, D=[])
                            print cut
                            #_["tokens"] = toks
                            #of.write(json.dumps(_) + "\n")

save_split('preproc/lstm_train.jsonl', train)

save_split('preproc/lstm_validation.jsonl', val)
