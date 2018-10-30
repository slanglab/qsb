# coding: utf-8

import json

import random

from code.treeops import prune
from code.treeops import dfs
from code.treeops import bfs
from code.treeops import get_walk_from_root
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


def get_labeled_toks(node, jdoc):
    toks = [i for i in jdoc["tokens"]]
    cut = dfs(g=jdoc, hop_s=node, D=[])
    cut.sort()
    mint = min(cut)
    maxt = max(cut)
    assert len(cut) == len(range(mint, maxt + 1))
    labeled_toks = []
    for counter, t in enumerate(toks):
        if t["index"] == mint:
            labeled_toks.append(START)
        labeled_toks.append(t["word"])
        if t["index"] == maxt:
            labeled_toks.append(END)
    return labeled_toks


def is_prune_only(jdoc):
    one_extract = Counter(_["oracle"].values())["e"] == 1
    extract_v = int([k for k, v in _["oracle"].items()
                    if v == "e"][0])
    gov = [i["governor"] for i in _["basicDependencies"] if
           i["dependent"] == extract_v][0]
    return gov == 0 and one_extract


def save_split(fn, data, cap=None):
    '''
    note, avoiding pulling a whole big corpus into memory so this can scale up
    inputs:
        fn (str): output file
        data (list<int>): a list of lines that are included in training set
        cap (int): maximum number of examples to generate
    '''
    total_so_far = 0
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, _ in enumerate(inf):
                if cap is not None and total_so_far > cap:
                    break  # early stopping
                if ino in data:
                    _ = json.loads(_)
                    if is_prune_only(jdoc=_):
                        walk = [(node, depth) for node, depth in
                                get_walk_from_root(_) if depth > 0]
                        for node, depth in walk:
                            toks_remaining = [i["index"] for i in _["tokens"]]
                            oracle_label = _["oracle"][str(node)]

                            ## for now, let's just do binary classification
                            ## This extract op does not work in obvious ways
                            ## w/ iterative deletion as extract adds tokens to
                            ## the buffer. Another view of this is bottom up where
                            ## the decision is "attach" or "finish". That might
                            ## be an easier way to unify prune and extract
                            if oracle_label == "e":
                                oracle_label == "NA"
                            if node in toks_remaining:
                                tmp = {
                                    "compression_indexes": _["compression_indexes"],
                                    "label": _["oracle"][str(node)],
                                    "tokens": get_labeled_toks(node, _)
                                }
                                of.write(json.dumps(tmp) + "\n")
                                total_so_far += 1
                                if tmp["label"] == "p":
                                    prune(g=_, v=node)

save_split('preproc/lstm_train.jsonl', train)

save_split('preproc/lstm_validation.jsonl', val, 10000)
