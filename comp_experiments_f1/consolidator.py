import glob
import json
import json
import numpy as np
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))


def get_sentences(dt):
    return [k for k in dt.keys() if "sentence" in k]

def has_lm(dt):
    return "lm" in dt[get_sentences(dt)[0]]

for fn in glob.glob("comp_experiments_f1/output/*"):
    if "worst" not in fn:
        with open(fn, "r") as inf:
            dt = json.load(inf)
            if has_lm(dt): 
                arch = dt["archive_loc"].replace("model.tar.gz", "metrics.json")
                with open(arch, "r") as inf:
                    metrics =json.load(inf)
                metrics["f1"] = np.mean([float(dt[v]["f1"]) for v in dt.keys() if "sentence" in v])
                metrics["archive"] = dt["archive_loc"]
                metrics['algo'] = dt['algorithm']
                metrics['lm'] = np.mean([float(dt[v]["lm"]) for v in dt.keys() if "sentence" in v])
                results[dt["algorithm"] + dt["archive_loc"]] = metrics
                print(fn, metrics["f1"])

with open("output/results.csv", "w") as of:
    first = True
    for ln in results:
        ln = results[ln]
        if first:
            of.write(",".join(ln.keys()) + "\n")
            first = False
        v = [str(ln[o]) for o in ["algo", "f1", "lm"]]
        of.write(",".join(v) + "\n")
