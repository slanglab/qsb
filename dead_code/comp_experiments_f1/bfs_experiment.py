import json
import csv

from sklearn.metrics import f1_score
from code.treeops import bfs
from tqdm import tqdm
from singleop.predictors import FigureEightPredictor

mini_validation_set = []

predictor = FigureEightPredictor(cache="cache/")

with open("preproc/training.jsonl", "r") as inf:
    for vno, _ in enumerate(inf):
        mini_validation_set.append(json.loads(_))
        if vno > 250:
            break


def get_p_endorsement_v(v, jdoc):
    dep = [_["dep"] for _ in jdoc["basicDependencies"]
           if int(_["dependent"]) == int(v)][0]
    return predictor.predict_proba(jdoc=jdoc,
                                   op="prune",
                                   vertex=int(v),
                                   dep=dep,
                                   worker_id=0
                                   )

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


def get_root(w):
    return [_["dependent"] for _ in w["basicDependencies"]
            if _["dep"].lower() == "root"][0]

T = .40
from code.treeops import prune

f1s = []
ns = []

for w in tqdm(mini_validation_set):
    d, pi, c = bfs(g=w, hop_s=get_root(w))
    nodes_depths = d.items()
    nodes_depths.sort(key=lambda x:x[1])
    nops = 0
    gold = [_["index"] in w["compression_indexes"] for _ in w["tokens"]] 
    orig = [_["index"] for _ in w["tokens"]] 
    for node, depth in nodes_depths:
        vertexes_remaining = [_["index"] for _ in w["tokens"]] 
        len_ = len(" ".join([_["word"] for _ in w["tokens"]]))
        if node in vertexes_remaining and len_ > w['r']:
            if depth > 0:
                nops += 1
                pe = get_p_endorsement_v(node, jdoc=w)
                if pe > T:
                    prune(g=w, v=node)

    included = [_["index"] for _ in w["tokens"]]
    pred = [_ in included for _ in orig]
    f1s.append(f1_score(y_true=gold, y_pred=pred)) 
    ns.append(nops)

import numpy as np
print np.mean(f1s)
