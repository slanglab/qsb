import sys
import json
from code.log import logger
from unidecode import unidecode
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform, get_tok

def get_lines(fn):
    with open(fn, "r") as inf:
        for ln in inf:
            yield ln

def get_toks(jdoc):
    for t in jdoc['tokens']:
        yield t

def get_deps(jdoc):
    for t in jdoc['enhancedDependencies']:
        yield t["dep"]

if __name__ == "__main__":

    V = set()
    dep_v = set(["ROOT"])
    pos_v = set(["ROOT"])
    ner_v = set(["O"])
    lemma_v = set(["ROOT"])
    lemma_v_dep_v = set(["ROOT-ROOT"])
    for ln in get_lines(sys.argv[1]):
        ln2 = filippova_tree_transform(json.loads(ln))
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
                logger.warning("this dep points to non existing token$${}$${}".format(ln, json.dumps(d))) 
        for d in get_deps(ln2):
            dep_v.add(unidecode(d))

    print "the size of lemma_v_dep_v is ", len(lemma_v_dep_v)

    with open(sys.argv[1] + ".vocabs", "w") as of:
        of.write(json.dumps({"V":list(V), "pos_v": list(pos_v), "dep_v":list(dep_v), "ner_v":list(ner_v), "lemma_v_dep_v": list(lemma_v_dep_v), "lemma_v":list(lemma_v)}))
