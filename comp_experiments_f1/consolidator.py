import glob
import numpy as np

models = ["nn"]

for fn in glob.glob("comp_experiments_f1/output/*"):
    with open(fn, "r") as inf:
        dt = inf.read()
        f1s = []
        nops = []
        for k, v in dt.items():
            if "sentence" in k:
                f1s.append(float(v["f1"]))
                nops.append(float(v["nops"]))
        f1s = np.mean(f1s)
        ops = np.mean(nops)
        print f1s, nops