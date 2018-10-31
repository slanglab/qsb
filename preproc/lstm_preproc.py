# coding: utf-8

import json

import random

from code.treeops import prune
from code.treeops import get_walk_from_root
from collections import Counter
from code.utils import get_labeled_toks


def get_root(w):
    return [_["dependent"] for _ in w["basicDependencies"]
            if _["dep"].lower() == "root"][0]


def is_prune_only(jdoc):
    one_extract = Counter(jdoc["oracle"].values())["e"] == 1
    extract_v = int([k for k, v in jdoc["oracle"].items()
                    if v == "e"][0])
    gov = [i["governor"] for i in jdoc["basicDependencies"] if
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
                        walk = get_walk_from_root(_)
                        for node in walk:
                            toks_remaining = [i["index"] for i in _["tokens"]]
                            oracle_label = _["oracle"][str(node)]
                            dep = [ii["dep"] for ii in _["basicDependencies"]
                                   if _["dependent"] == node][0]
                            ## for now, let's just do binary classification
                            ## This extract op does not work in obvious ways
                            ## w/ iterative deletion as extract adds tokens to
                            ## the buffer. Another view of this is bottom up where
                            ## the decision is "attach" or "finish". That might
                            ## be an easier way to unify prune and extract
                            if oracle_label == "e":
                                oracle_label = "NA"

                            if node in toks_remaining:
                                tmp = {
                                    "compression_indexes": _["compression_indexes"],
                                    "label": oracle_label,
                                    "dep": dep,
                                    "tokens": get_labeled_toks(node, _)
                                }
                                of.write(json.dumps(tmp) + "\n")
                                total_so_far += 1
                                if tmp["label"] == "p":
                                    prune(g=_, v=node)


if __name__ == "__main__":

    c = 0

    PCT_TRAIN = .95

    CORPUS = 'preproc/training.jsonl'

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

    save_split('preproc/lstm_train.jsonl', train, 100000)

    save_split('preproc/lstm_validation.jsonl', val, 10000)
