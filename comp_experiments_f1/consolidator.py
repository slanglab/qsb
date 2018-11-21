import glob
import json
import numpy as np
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))


for fn in glob.glob("comp_experiments_f1/output/*"):
    with open(fn, "r") as inf:
        dt = json.load(inf)
        results[dt["algorithm"] + dt["archive_loc"]] = np.mean([float(dt[v]["f1"]) for v in dt.keys() if "sentence" in v])
        print([o for o in dt.keys() if "sente" not in o]) 
        results["archive"] = dt["archive_loc"]

for model in results:
    print("***")
    print(model)
    print(results[model]) 


def sigfigs(val, figs="3"):
    return str("{0:1." + figs + "f}").format(val)

import os;os._exit(0)

with open("latex/f1.tex", "w") as of:

    # min compression
    f1_only_q = sigfigs(results["min-compression"])
    out = "Query terms only & {}    \\\\ ".format(f1_only_q)
    of.write(out + "\n")

    # prune only greedy
    f1_greedy_prune = sigfigs(results["nn-prune-greedy"])
    out = " \\textbf{{Iterative deletion}} &  \\textbf{{{}}}    \\\\ ".format(f1_greedy_prune)
    of.write(out + "\n")

