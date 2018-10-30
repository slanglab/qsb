# coding: utf-8

import json

import random

from code.treeops import prune
from code.treeops import dfs
from code.treeops import bfs
from collections import Counter

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


def get_root(w):
    return [_["dependent"] for _ in w["basicDependencies"]
            if _["dep"].lower() == "root"][0]


def get_labeled_toks(mint, maxt, toks):
    labeled_toks = []
    for counter, t in enumerate(toks):
        if t["index"] == mint:
            labeled_toks.append(START)
        labeled_toks.append(t["word"])
        if t["index"] == maxt:
            labeled_toks.append(END)
    return labeled_toks

def save_split(fn, data, cap=None):
    '''note, avoiding pulling a whole big corpus into memory so this can scale up'''
    total_so_far = 0
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, _ in enumerate(inf):
                if cap is not None and total_so_far > cap:
                    break  # early stopping
                if ino in data:
                    _ = json.loads(_)
                    oracle = _["oracle"]
                    one_extract = Counter(oracle.values())["e"] == 1 
                    if one_extract:
                        extract_v = int([k for k,v in _["oracle"].items() if v == "e"][0])
                        gov = [i["governor"] for i in _["basicDependencies"] if i["dependent"] == extract_v][0]
                         
                        del _["index"]
                        del _["enhancedDependencies"]
                        del _["enhancedPlusPlusDependencies"]
                        toks = [i for i in _["tokens"]]
                        d, pi, c = bfs(g=_, hop_s=get_root(_))
                        nodes_depths = d.items()
                        nodes_depths.sort(key=lambda x: x[1])

                        for node, depth in nodes_depths:
                            if depth > 0:
                                toks_remaining = [i["index"] for i in _["tokens"]]
                                if node in toks_remaining:
                                    oracle = _["oracle"][str(node)]
                                    cut = dfs(g=_, hop_s=node, D=[])
                                    cut.sort()
                                    mint = min(cut)
                                    maxt = max(cut)
                                    assert len(cut) == len(range(mint, maxt + 1))
                                    labeled_toks = get_labeled_toks(mint, maxt, toks)
                                    tmp = {k: v for k, v in _.items() if k in
                                           ["tokens", "label", "compression_indexes"]}
                                    tmp["tokens"] = labeled_toks
                                    tmp["label"] = oracle
                                    of.write(json.dumps(tmp) + "\n")
                                    total_so_far += 1
                                    if oracle == "p":
                                        prune(g=_, v=node)

save_split('preproc/lstm_train.jsonl', train)

save_split('preproc/lstm_validation.jsonl', val, 10000)
