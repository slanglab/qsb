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


def save_split(fn, data, cap=None):
    '''note, avoiding pulling a whole big corpus into memory so this can scale up'''
    total_so_far = 0
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, _ in enumerate(inf):
                if cap is not None and total_so_far > cap:
                    break # early stopping
                if ino in data:
                    _ = json.loads(_) 
                    
                    del _["index"]
                    del _["enhancedDependencies"]
                    del _["enhancedPlusPlusDependencies"]
                    toks = [i for i in _["tokens"]]
                    for vertex in _["tokens"]:
                        oracle = _["oracle"][str(vertex["index"])]
                        _['label'] = oracle
                        cut = dfs(g=_, hop_s = vertex["index"], D=[])
                        cut.sort()
                        mint = min(cut)
                        maxt = max(cut)
                        assert len(cut) == len(range(mint, maxt + 1))
                        labeled_toks = []
                        if len(cut) < len(toks): 
                            for counter, t in enumerate(toks):  
                                if t["index"] == mint:
                                    labeled_toks.append(START)
                                labeled_toks.append(t["word"])
                                if t["index"] == maxt:
                                    labeled_toks.append(END) 
                            _["tokens"] = labeled_toks 
                            tmp = {k:v for k,v in _.items() if k in ["tokens", "label", "original", "compression_indexes"]}
                            of.write(json.dumps(tmp) + "\n")
                            total_so_far += 1 

save_split('preproc/lstm_train.jsonl', train)

save_split('preproc/lstm_validation.jsonl', val, 10000)
