# coding: utf-8

import json
import copy
import random

from code.treeops import prune
from code.treeops import get_walk_from_root
from code.treeops import extract_for_state
from collections import Counter
from code.utils import get_labeled_toks
from code.printers import pretty_print_conl


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


def save_split(fn, data, threeway=False, cap=None):
    '''
    note, avoiding pulling a whole big corpus into memory so this can scale up
    inputs:
        fn (str): output file
        data (list<int>): a list of lines that are included in training set
        cap (int): maximum number of examples to generate
        threeway (bool): do 3-way classification?
    '''
    total_so_far = 0
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, _ in enumerate(inf):
                if cap is not None and total_so_far > cap:
                    break  # early stopping
                if ino in data:
                    print("***")
                    print(ino)
                    _ = json.loads(_)
                    se_ve = copy.deepcopy(_)
                    orig_ix = [i["index"] for i in _["tokens"]]
                    r = " ".join([i["word"] for i in _["tokens"] if i["index"]
                                  in _["compression_indexes"]])
                    r = len(r)
                    deps = copy.deepcopy(_["basicDependencies"])
                    if threeway or (not threeway and is_prune_only(jdoc=_)):
                        walk = get_walk_from_root(_)
                        state = {"tokens": [], "basicDependencies": []}
                        for node in walk:
                            toks_remaining = [i["index"] for i in _["tokens"]]
                            oracle_label = _["oracle"][str(node)]
                            print(oracle_label)
                            ## for now, let's just do binary classification
                            ## This extract op does not work in obvious ways
                            ## w/ iterative deletion as extract adds tokens to
                            ## the buffer. Another view of this is bottom up where
                            ## the decision is "attach" or "finish". That might
                            ## be an easier way to unify prune and extract
                            if oracle_label == "e" and not threeway:
                                oracle_label = "NA"

                            # current vertexes in compression
                            vc = [t["index"] for t in state]
                            if (oracle_label == "e") or (node in vc):
                                dep = [ii["dep"] for ii in _["basicDependencies"]
                                       if ii["dependent"] == node][0]
                                tmp = {
                                    "compression_indexes": _["compression_indexes"],
                                    "label": oracle_label,
                                    "dep": dep,
                                    "tokens": get_labeled_toks(node, state),
                                    "q": _['q'],
                                    "r": r,
                                    "original_ix": orig_ix,
                                    "basicDependencies": deps
                                }
                                of.write(json.dumps(tmp) + "\n")
                                total_so_far += 1
                                if tmp["label"] == "p":
                                    prune(g=state, v=node)
                                if tmp["label"] == "e":
                                    subtree = extract_for_state(g=se_ve, v=node)
                                    state["tokens"] = state["tokens"] + subtree["tokens"]
                                    state["basicDependencies"] = state["basicDependencies"] + subtree["basicDependencies"]

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

    N = 1000000

    save_split('preproc/lstm_train_3way.jsonl', train, cap=N, threeway=True)
    save_split('preproc/lstm_validation_3way.jsonl', val, cap=10000, threeway=True)

    save_split('preproc/lstm_train.jsonl', train, cap=N, threeway=False)
    save_split('preproc/lstm_validation.jsonl', val, cap=10000, threeway=False)
