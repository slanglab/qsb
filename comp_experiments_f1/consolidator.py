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
with open("output/results.csv", "w") as of:
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
