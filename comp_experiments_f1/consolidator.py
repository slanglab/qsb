import glob
import json
import numpy as np
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))


for fn in glob.glob("comp_experiments_f1/output/*"):
    with open(fn, "r") as inf:
        dt = json.load(inf)
        f1s_prunes = []
        for k, v in dt.items():
            if "sentence" in k:
                results[dt["algorithm"]]["f1"].append(v["f1"])
                results[dt["algorithm"]]["nops"].append(v["nops"]) 
                f1s_prunes.append((v["f1"], v["prunes"]))
            results[dt["algorithm"]]["no_compression"] = dt["no_compression"]

with open("/tmp/f1vprunes.txt", "w") as of:
    for _ in f1s_prunes:
        f1, p = _
        of.write("{},{}\n".format(f1, p))

for model in results:
    print(model)
    print(results[model]["no_compression"])
    print(np.mean(results[model]["f1"]))
    print(np.mean(results[model]["nops"]))


def sigfigs(val, figs="3"):
    return str("{0:1." + figs + "f}").format(val)


with open("latex/f1.tex", "w") as of:

    # min compression
    f1_only_q = sigfigs(np.mean(results["min-compression"]["f1"]))
    out = "Query terms only & {}    \\\\ ".format(f1_only_q)
    of.write(out + "\n")


    # prune only greedy
    f1_greedy_prune = sigfigs(np.mean(results["nn-prune-greedy"]["f1"]))
    out = " \\textbf{{Iterative deletion}} &  \\textbf{{{}}}    \\\\ ".format(f1_greedy_prune)
    of.write(out + "\n")

