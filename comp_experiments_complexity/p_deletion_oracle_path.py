import json
import csv

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


cats = []
for w in tqdm(mini_validation_set):
    probs_endorse = p_endorsement(w)
    w["p_endorsement"] = probs_endorse
    for v in probs_endorse:
        p_yes = probs_endorse[v]
        oracle = w["oracle"][str(v)]
        if oracle == "NA":
            oracle = "k"
        cats.append((oracle, str(p_yes)))

with open('output/probs_oracles.csv', "w") as of:
    writer = csv.writer(of)
    writer.writerow(['oracle', 'p'])
    writer.writerows(cats)
