import json
import re

from sklearn.metrics import f1_score
from code.treeops import dfs
from code.treeops import prune
from code.printers import pretty_print_conl
from singleop.predictors import FigureEightPredictor
from code.utils import get_ner_spans_in_compression, get_min_compression


with open("tests/fixtures/pakistan_example.json", "r") as inf:
    dt = json.load(inf)["sentences"][0]

predictor = FigureEightPredictor(cache="cache/")

pretty_print_conl(dt)

print get_min_compression(8, dt)

validation_set = []


with open("preproc/validation.jsonl", "r") as inf:
    for vno, _ in enumerate(inf):
        validation_set.append(json.loads(_))
        if vno > 1000:
            break


QSRs = []
for s in validation_set:
    print "***"
    q = [i for _ in get_ner_spans_in_compression(s) for i in _]
    r = len(" ".join([i["word"] for i in s["tokens"] if
                      i['index'] in s["compression_indexes"]]))
    if len(q) > 0:
        QSRs.append((set(q), s, r))


def p_endorsement(jdoc):
    out = {}
    for tok in jdoc["tokens"]:
        v = int(tok["index"])
        dep = [_["dep"] for _ in jdoc["basicDependencies"]
               if int(_["dependent"]) == int(v)][0]
        # jdoc, op, vertex, dep, worker_id=0
        if dep.lower() != "root":
            out[v] = predictor.predict_proba(jdoc=jdoc,
                                             op="prune",
                                             vertex=v,
                                             dep=dep,
                                             worker_id=0
                                             )
    return out


def len_s(jdoc):
    return len(" ".join([_["word"] for _ in jdoc["tokens"]]))


def deletes_q(v, q, jdoc):
    '''if you were to prune this vertex, would you remove any v \in q?'''
    children = set(dfs(g=jdoc, hop_s=v, D=[]))
    return len(q & children) > 0


def greedy_humans(qsr):
    q, s, r = qsr
    orig_toks = [_["index"] for _ in s["tokens"]]
    y_true = [_ in s['compression_indexes'] for _ in orig_toks] 
    while len_s(s) > r:
        probs = [(vertex, prob) for vertex, prob in p_endorsement(jdoc=s).items()
                 if not deletes_q(v=vertex, q=q, jdoc=s)]
        probs.sort(key=lambda x: x[0], reverse=True)
        if len(probs) == 0:
            print "unable to compress"
            break
        best_v, best_p = probs[0]
        prune(g=s, v=best_v)
    c_ix = [_["index"] for _ in s['tokens']]
    y_pred = [_ in c_ix for _ in orig_toks]
    print f1_score(y_true, y_pred)

for wno, w in enumerate(QSRs):
    print wno
    greedy_humans(w)
