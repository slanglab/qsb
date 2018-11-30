import glob
import json
import numpy as np
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))


for fn in glob.glob("comp_experiments_f1/output/*"):
    if "worst" not in fn:
        with open(fn, "r") as inf:
            dt = json.load(inf)
            arch = dt["archive_loc"].replace("model.tar.gz", "metrics.json")
            with open(arch, "r") as inf:
                metrics =json.load(inf)
            metrics["f1"] = np.mean([float(dt[v]["f1"]) for v in dt.keys() if "sentence" in v])
            metrics["archive"] = dt["archive_loc"]
            metrics['algo'] = dt['algorithm']
            results[dt["algorithm"] + dt["archive_loc"]] = metrics
            print(fn, metrics["f1"])

import json
with open("/tmp/opt.csv", "w") as of:
    first = True
    for ln in results:
        ln = results[ln]
        if first:
            of.write(",".join(ln.keys()) + "\n")
            first = False
        v = [str(ln[o]) for o in ln.keys()]
        of.write(",".join(v) + "\n")   

with open("/tmp/optimize.jsonl", "w") as of:
    for r in results:
        of.write(json.dumps(results[r]) + "\n")

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

