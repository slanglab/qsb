import glob
import numpy as np

models = ["nn"]
for mo in models:
    all_ = []
    for fn in glob.glob("comp_experiments_f1/output/*-nn"):
        with open(fn, "r") as inf:
            dt = inf.read()
            all_.append(float(dt))
    print np.mean(all_) 
