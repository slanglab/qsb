import glob
import json
import json
import numpy as np
from collections import defaultdict

results = [] 

def get_sentences(dt):
    return [k for k in dt.keys() if "sentence" in k]

def has_lm(dt):
    return "lm" in dt[get_sentences(dt)[0]]

for fn in glob.glob("comp_experiments_f1/output/*"):
#    if "worst" not in fn:
    with open(fn, "r") as inf:
        dt = json.load(inf)
        if has_lm(dt): 
            arch = dt["archive_loc"].replace("model.tar.gz", "metrics.json")
            metrics = {}
            metrics["f1"] = np.mean([float(dt[v]["f1"]) for v in dt.keys() if "sentence" in v])
            metrics["archive"] = dt["archive_loc"]
            metrics['algo'] = dt['algorithm']
            metrics['lm'] = np.mean([float(dt[v]["lm"]) for v in dt.keys() if "sentence" in v])
            metrics["fn"] = fn
            print(fn, metrics["f1"], fn)
            results.append(metrics)

def write_results(kind, results):
    with open("output/{}.csv".format(kind), "w") as of:
        first = True
        for ln in results:
            if first:
                of.write(",".join(["algo", "f1", "lm", "fn"]) + "\n")
                first = False
            v = [str(ln[o]) for o in ["algo", "f1", "lm", "fn"]]
            of.write(",".join(v) + "\n")


# write the results for assessing ILP convergence, discussed in the appendix
write_results("ilp_convergence", [o for o in results if "validation" in o["fn"] and "ilp" in o["fn"] and "full" in o["fn"]])

 
write_results("test", [o for o in results if "validation" not in o["fn"] and "full" in o["fn"]]) 
