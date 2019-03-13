from tqdm import tqdm
import numpy as np
import json
import random
import pickle
import argparse
import timeit
from ilp2013.fillipova_altun import run_model
from ilp2013.fillipova_altun_supporting_code import get_all_vocabs

parser = argparse.ArgumentParser()
parser.add_argument('-path_to_set_to_evaluate', type=str, default="validation.paths")

args = parser.parse_args()

S = []

weights = "snapshots/8"

with open(weights, "rb") as of:
    weights = pickle.load(of)

vocabs = get_all_vocabs()

with open(args.path_to_set_to_evaluate, "r") as inf:
    for ln in inf:
        ln = json.loads(ln)
        S.append(ln)


def test_ILP():
    """Do compression"""
    s = random.sample(S, k=1)[0]["sentence"]
    run_model(s, r=s["r"], Q=s["q"], vocab=vocabs, weights=weights) 


if __name__ == '__main__':

    all_ = []
    for i in tqdm(range(10000)):
        a = timeit.timeit("test_ILP()", setup="from __main__ import test_ILP", number=1)
        all_.append(a)
    print(np.mean(all_), np.std(all_))
