'''
- Make a validation and training set from the processed data
- Also make the vocabs for the ilp
- Also determine a q and r based on NER in gold compression
- Also create an oracle path
'''

import pickle
import random
import json
import glob

from code.log import logger
from unidecode import unidecode
from ilp2013.fillipova_altun_supporting_code import get_tok
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform
from code.utils import get_ner_spans_in_compression
from code.treeops import find_maximal_subtrees

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


def load_dataset():
    sources = []

    for source in glob.glob("sentence-compression/data/*sent-comp*source"):
        with open(source, "r") as inf:
            for ln in inf:
                ln = json.loads(ln)
                q = [i for _ in get_ner_spans_in_compression(ln) for i in _]
                r = len(" ".join([i["word"] for i in ln["tokens"] if
                        i['index'] in ln["compression_indexes"]]))
                ln["r"] = r
                ln["q"] = q
                trees = find_maximal_subtrees(copy.deepcopy(ln), ln["compression_indexes"])
                oracle = get_oracle(copy.deepcopy(ln), ln["compression_indexes"], trees)
                sources.append(ln)
    return sources


if __name__ == "__main__":
    data = load_dataset()
    random.shuffle(data)
    data = [d for d in data if len(d["q"]) > 0]
    print "[*] dataset loaded"

    # this is for validation. note: no tree transform.
    # this will go to human experiments
    # the prune based models do not use transforms so the transform happens there
    with open("preproc/validation.jsonl", "w") as of:
        dt = "\n".join([json.dumps(_) for _ in data[-validation_size:]])
        of.write(dt)

    print "[*] dumped validation examples"
    # this is to train lstm taggers 
    with open("preproc/training.jsonl", "w") as of:
        dt = [json.dumps(_) for _ in data[0:-validation_size]]
        print len(dt)
        of.write("\n".join(dt))

    print "[*] dumped validation examples"
    # this is to train the ILP from F & A
    with open("preproc/100k", "w") as of:
        dt = [filippova_tree_transform(json.loads(i)) for i in dt[0:100000]] 
        pickle.dump(dt, of)

    print "[*] building vocab"

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
            assert type(t["pos"]) != str
            ner_v.add(unidecode(t['ner']))
            assert type(t["ner"]) != str
            lemma_v.add(unidecode(t['lemma']))
            assert type(t["lemma"]) != str
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
