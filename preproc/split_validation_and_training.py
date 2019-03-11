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

from bottom_up_clean.query_maker import get_q
from code.log import logger
from unidecode import unidecode
from ilp2013.fillipova_altun_supporting_code import get_tok
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
            for ln in inf:
                ln = json.loads(ln)
                r = len(" ".join([i["word"] for i in ln["tokens"] if
                        i['index'] in ln["compression_indexes"]]))
                ln["r"] = r
                ln["q"] = get_q(ln)
                sources.append(ln)
    return sources


if __name__ == "__main__":
    data = load_dataset()
    from random import Random
    Random(42).shuffle(data)
    data = [d for d in data if len(d["q"]) > 0]

    # this is for validation. note: no tree transform.
    # this will go to human experiments
    # the prune based models do not use transforms so the transform happens there
    with open("preproc/validation.jsonl", "w") as of:
        dt = "\n".join([json.dumps(_) for _ in data[-validation_size:]])
        of.write(dt)

    # this is to train lstm taggers 
    with open("preproc/training.jsonl", "w") as of:
        dt = [json.dumps(_) for _ in data[0:-validation_size]]
        of.write("\n".join(dt))

    # this is to train the ILP from F & A
    with open("preproc/100k", "wb") as of:
        dt = [filippova_tree_transform(json.loads(i)) for i in dt[0:100000]] 
        pickle.dump(dt, of)

    def get_toks(jdoc):
        for t in jdoc['tokens']:
            yield t

    def get_deps(jdoc):
        for t in jdoc['enhancedDependencies']:
            yield t["dep"]

    V = set()
    dep_v = set(["ROOT"])
    pos_v = set(["ROOT"])
    ner_v = set(["O"])
    lemma_v = set(["ROOT"])
    lemma_v_dep_v = set(["ROOT-ROOT"])
    for ln2 in dt:
        for t in get_toks(ln2):
            V.add(t["word"])
            pos_v.add(unidecode(t["pos"]))
            ner_v.add(unidecode(t['ner']))
            lemma_v.add(unidecode(t['lemma']))
        for d in ln2["enhancedDependencies"]:
            try:
                lemma = get_tok(d["governor"], jdoc=ln2)["lemma"]
                lemma_v_dep_v.add(unidecode(lemma + "-" + d["dep"]))
            except:
                logger.warning("this dep points to non existing token$${}$${}".format(json.dumps(ln2), json.dumps(d)))
        for d in get_deps(ln2):
            dep_v.add(unidecode(d))

    with open("preproc/vocabs", "w") as of:
        of.write(json.dumps({"V": list(V),
                             "pos_v": list(pos_v),
                             "dep_v": list(dep_v),
                             "ner_v": list(ner_v),
                             "lemma_v_dep_v": list(lemma_v_dep_v),
                             "lemma_v": list(lemma_v)}))
