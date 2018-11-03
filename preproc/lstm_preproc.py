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


def save_split_3way(fn, data, cap=None):
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
                    se_ve = copy.deepcopy(_)
                    orig_ix = [i["index"] for i in _["tokens"]]
                    r = len(" ".join([i["word"] for i in _["tokens"] if i["index"]
                                     in _["compression_indexes"]]))
                    walk = get_walk_from_root(_)
                    state = {"tokens": [], "basicDependencies": []}
                    for node in walk:
                        oracle_label = _["oracle"][str(node)]
                        if oracle_label == "NA":
                            if node in walk:
                                oracle_label = "np"  # no prune
                            else:
                                oracle_label = "ne"  # no extract

                        dep = [ii["dep"] for ii in _["basicDependencies"]
                               if ii["dependent"] == node][0]

                        encoding = {
                            "compression_indexes": _["compression_indexes"],
                            "label": oracle_label,
                            "dep": dep,
                            "q": _['q'],
                            "r": r,
                            "original_ix": orig_ix,
                            "basicDependencies": se_ve["basicDependencies"]
                        }

                        if oracle_label in ["p", "np"]:
                            encoded_tokens = get_labeled_toks(node, state, "p")
                        elif oracle_label in ["e", "ne"]:
                            proposed = extract_for_state(g=se_ve, v=node)
                            proposed["tokens"] = state["tokens"] + proposed["tokens"]
                            proposed["basicDependencies"] = state["basicDependencies"] + proposed["basicDependencies"]
                            encoded_tokens = get_labeled_toks(node, proposed, "e")
                        else:
                            assert "bad" == "thing"

                        encoded_tokens.sort(key=lambda x: float(x["index"]))
                        encoding["tokens"] = encoded_tokens

                        of.write(json.dumps(encoding) + "\n")
                        total_so_far += 1
                        if encoding["label"] == "p":
                            prune(g=state, v=node)
                        if encoding["label"] == "e":
                            state["tokens"] = proposed["tokens"]
                            state["basicDependencies"] = proposed["basicDependencies"]
                transition = [o["index"] for o in state["tokens"]]
                assert set(transition) == set(_["compression_indexes"])

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

    save_split_3way('preproc/lstm_train_3way.jsonl', train, cap=N)
    save_split_3way('preproc/lstm_validation_3way.jsonl', val, cap=10000)

    #save_split('preproc/lstm_train.jsonl', train, cap=N, threeway=False)
    #save_split('preproc/lstm_validation.jsonl', val, cap=10000, threeway=False)
