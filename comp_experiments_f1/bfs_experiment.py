import json
import csv

from code.treeops import bfs
from tqdm import tqdm
from singleop.predictors import FigureEightPredictor

mini_validation_set = []

predictor = FigureEightPredictor(cache="cache/")

with open("preproc/training.jsonl", "r") as inf:
    for vno, _ in enumerate(inf):
        mini_validation_set.append(json.loads(_))
        if vno > 1000:
            break


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


for w in tqdm(mini_validation_set):
    d, pi, c = bfs(g=w, hop_s=get_root(w))
    nodes_depths = d.items()
    nodes_depths.sort(key=lambda x:x[1])
    print nodes_depths