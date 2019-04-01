'''
- Make a validation and training set from the processed data
- Also make the vocabs for the ilp
- Also determine a q and r
- Also create an oracle path
'''

import pickle
import random
import json
import glob
import copy

from tqdm import tqdm
from bottom_up_clean.query_maker import get_q
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform

random.seed(1)

validation_size = 25000


def get_oracle(sentence, compresion_ixs, trees):
    '''
    get the oracle move for each token in the sentence
    inputs:
        sentence (dict):  a jdoc sentence
        compresion_ixs (list:int): the indexes of the compressed tokens
    '''

    pruned = [i for t in trees for i in t.pruned]
    extracted = [i.root for i in trees]
    def oracle_move(v):
        if v in pruned:
            return "p"
        if v in extracted:
            return "e"
        return "NA"

    oracle = {i["index"]: oracle_move(i["index"])
              for i in sentence["tokens"]}
    return oracle


# important. glob_string for the *training data* starts sent-comp. test data is comp-data-eval in same folder. glob won't get it by default, which is important as otherwise you are training on test ! 
def load_dataset(glob_string="sentence-compression/data/sent-comp*source"):
    sources = []

    for source in glob.glob(glob_string):
        with open(source, "r") as inf:
            for ln in tqdm(inf):
                try:
                    ln = json.loads(ln)
                    r = len(" ".join([i["word"] for i in ln["tokens"] if
                            i['index'] in ln["compression_indexes"]]))
                    ln["r"] = r
                    ln["q"] = list(get_q(ln))
                    sources.append(ln)
                except AssertionError:
                    pass  # in a rare cases (~1/1000) there is no possible Q
    return sources


if __name__ == "__main__":
    data = load_dataset()
    from random import Random
    Random(42).shuffle(data)
    data = [d for d in data if len(d["q"]) > 0]

    # this is for validation. note: no tree transform.
    # the prune based models do not use transforms so the transform happens there
    with open("preproc/validation.jsonl", "w") as of:
        dt = "\n".join([json.dumps(_) for _ in data[-validation_size:]])
        of.write(dt)

    # this is to train stateful models
    with open("preproc/training.jsonl", "w") as of:
        dt = [json.dumps(_) for _ in data[0:-validation_size]]
        of.write("\n".join(dt))

    # this is to train the ILP from F & A
    with open("preproc/training.ilp", "wb") as of:
        dt = [filippova_tree_transform(json.loads(i)) for i in dt]
        pickle.dump(dt, of)
