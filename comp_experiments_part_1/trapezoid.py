# coding: utf-8

# example useage $python comp_experiments_part_1/trapezoid.py comp_experiments_f1/output/fast-813281894-nn-prune-greedy 

import json

import sys

fn = sys.argv[1]


## fn example "comp_experiments_f1/output/fast-813281894-nn-prune-greedy"

with open(fn, "r") as inf:
    dt = json.load(inf)
    dt = [v["nops"] for k,v in dt.items() if "sentence" in k]

out = fn.split("/")[-1] + "trapezoid.csv"

max_ = max([len(i) for i in dt])


with open("output/" + out, "w") as of:
    of.write("sentence,epoch,ops\n")
    for sno, s in enumerate(dt):
        for opno, nops in enumerate(s):
            of.write("{},{},{}\n".format(sno,opno,nops)) 

#print(dt)
