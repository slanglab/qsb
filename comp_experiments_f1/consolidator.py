import glob
import json
import numpy as np
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))


for fn in glob.glob("comp_experiments_f1/output/*"):
    with open(fn, "r") as inf:
        dt = json.load(inf)
        f1s = []
        nops = []
        for k, v in dt.items():
            if "sentence" in k:
                results[dt["model"]]["f1"].append(v["f1"])
                results[dt["model"]]["nops"].append(v["nops"])
            results[dt['model']]["no_compression"] = dt["no_compression"]

for model in results:
    print(results[model]["no_compression"])
    print(np.mean(results[model]["f1"]))
    print(np.mean(results[model]["nops"]))
