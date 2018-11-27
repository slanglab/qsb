'''just a tmp workaround to fix an issue w/ preproc for ilp'''

from preproc import *
from unidecode import unidecode
from ilp2013.fillipova_altun_supporting_code import get_tok

from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform
import pickle
import json

dt = []

with open("preproc/training.jsonl", "r") as inf:
    ino = 0
    for i in inf:
        while(ino < 100000):
            ino += 1
            dt.append([filippova_tree_transform(json.loads(i))])


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
    toks = get_toks(ln2)
    import ipdb;ipdb.set_trace()
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
            print("error")
    for d in get_deps(ln2):
        dep_v.add(unidecode(d))

with open("preproc/vocabs", "w") as of:
    of.write(json.dumps({"V": list(V),
                         "pos_v": list(pos_v),
                         "dep_v": list(dep_v),
                         "ner_v": list(ner_v),
                         "lemma_v_dep_v": list(lemma_v_dep_v),
                         "lemma_v": list(lemma_v)}))
