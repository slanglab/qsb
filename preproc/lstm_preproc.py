# coding: utf-8

import json
import copy
import random

from code.treeops import prune
from code.treeops import get_walk_from_root
from code.treeops import extract_for_state
from collections import Counter
from code.utils import get_labeled_toks

PP = "π"  # "proposedprune"
PE = "ε"  # 'proposedextract'


def get_root(w):
    return [_["dependent"] for _ in w["basicDependencies"]
            if _["dep"].lower() == "root"][0]


def get_r(jdoc):
    return len(" ".join([i["word"] for i in jdoc["tokens"] if i["index"]
               in jdoc["compression_indexes"]]))


def get_oracle_label(jdoc, node, state):
    oracle_label = jdoc["oracle"][str(node)]
    vc = [i["index"] for i in state["tokens"]]
    if oracle_label == "NA":
        if node in vc:
            oracle_label = "np"  # no prune
        else:
            oracle_label = "ne"  # no extract
    return oracle_label 


def get_governing_dep(original_s, node):
    return [ii["dep"] for ii in original_s["basicDependencies"]
            if ii["dependent"] == node][0]


def get_instance(original_s, node, state):
    '''
    unknown oracle label is for test time
    '''
    orig_ix = [i["index"] for i in original_s["tokens"]]
    oracle_label = get_oracle_label(original_s, node, state)
    encoding = {
                "compression_indexes": original_s["compression_indexes"],
                "label": oracle_label,
                "dep": get_governing_dep(original_s, node),
                "q": original_s['q'],
                "r": get_r(original_s),
                "original_ix": orig_ix,
                "basicDependencies": original_s["basicDependencies"]
                }
    return encoding


def get_proposed(original_s, node, state):
    proposed = extract_for_state(g=original_s, v=node)
    proposed["tokens"] = state["tokens"] + proposed["tokens"]
    proposed["basicDependencies"] = state["basicDependencies"] + proposed["basicDependencies"]
    return proposed


def get_encoded_tokens(label, state, original_s, node):
    if label in ["p", "np"]:
        encoded_tokens = get_labeled_toks(node, state, PP)
    elif label in ["e", "ne"]:
        proposed = get_proposed(original_s, node, state)
        encoded_tokens = get_labeled_toks(node, proposed, PE)
    else:
        assert "bad" == "thing"
    return encoded_tokens


def save_split(fn, data, cap=None, keep_deps=False):
    '''
    note, avoiding pulling a whole big corpus into memory
    inputs:
        fn (str): output file
        data (list<int>): a list of lines that are included in training set
        cap (int): maximum number of examples to generate
    '''
    total_so_far = 0
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, original_s in enumerate(inf):
                if cap is not None and total_so_far > cap:
                    break  # early stopping
                if ino in data:
                    original_s = json.loads(original_s)
                    walk = get_walk_from_root(original_s)
                    state = {"tokens": [], "basicDependencies": []}
                    for node in walk:
                        instance = get_instance(original_s, node, state)
                        instance["tokens"] = get_encoded_tokens(instance["label"],
                                                                state,
                                                                original_s,
                                                                node)
                        # for faster load on gypsum
                        write_this = copy.deepcopy(instance)
                        if not keep_deps:
                            del write_this["basicDependencies"]
                        of.write(json.dumps(write_this) + "\n")

                        # update state if relevant
                        if instance["label"] == "p":
                            prune(g=state, v=node)
                        if instance["label"] == "e":
                            proposed = get_proposed(original_s, node, state)
                            state["tokens"] = proposed["tokens"]
                            state["basicDependencies"] = proposed["basicDependencies"]
                        total_so_far += 1

                    transition = [o["index"] for o in state["tokens"]]
                    assert set(transition) == set(original_s["compression_indexes"])


def just_save_sentences(fn, data, cap=None):
    '''
    note, avoiding pulling a whole big corpus into memory
    inputs:
        fn (str): output file
        data (list<int>): a list of lines that are included in training set
        cap (int): maximum number of examples to generate
    '''
    total_so_far = 0
    with open(CORPUS, 'r') as inf:
        with open(fn, 'w') as of:
            for ino, original_s in enumerate(inf):
                if cap is not None and total_so_far > cap:
                    break  # early stopping
                if ino in data:
                    s = json.loads(original_s)
                    s["original_ix"] = [i["index"] for i in s['tokens']]
                    of.write(json.dumps(s)  + "\n")
                    total_so_far += 1


if __name__ == "__main__":

    c = 0

    PCT_TRAIN = .95

    CORPUS = 'preproc/training.jsonl'

    with open(CORPUS, 'r') as inf:
        for _ in inf:
            c += 1

    c = list(range(c))
    random.shuffle(c)

    split_ix = int(len(c) * PCT_TRAIN)

    train = c[0:split_ix]
    test = c[split_ix:]

    val_ix = int((1 - PCT_TRAIN) * len(train))

    val = train[0:val_ix]
    train = train[val_ix:]

    N = 4000000

    save_split('preproc/lstm_train_{}.jsonl'.format(N), train, cap=N)

    save_split('preproc/lstm_validation_3way.jsonl', val, cap=10000, keep_deps=True)
    just_save_sentences('preproc/lstm_validation_sentences_3way.jsonl', val, cap=10000)
