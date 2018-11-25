'''just a tmp workaround to fix an issue w/ preproc for ilp'''

from preproc import *
from unidecode import unidecode
from ilp2013.fillipova_altun_supporting_code import get_tok
import pickle
import json

with open("preproc/100k", "r") as inf:
    dt = pickle.load(inf)


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
            print "error"
    for d in get_deps(ln2):
        dep_v.add(unidecode(d))

with open("preproc/vocabs", "w") as of:
    of.write(json.dumps({"V": list(V),
                         "pos_v": list(pos_v),
                         "dep_v": list(dep_v),
                         "ner_v": list(ner_v),
                         "lemma_v_dep_v": list(lemma_v_dep_v),
                         "lemma_v": list(lemma_v)}))