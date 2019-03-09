
from bottom_up_clean.all import *

from tqdm import tqdm
import json
import pickle

from bottom_up_clean.query_maker import get_q

def make_paths(fn):
    paths_loc = fn.replace(".jsonl", ".paths")

    failures = 0
    successes = 0
    with open(paths_loc, "w") as of:
        with open(fn, "r") as inf:
            for _ in tqdm(inf):
                s = json.loads(_)
                s["q"] = list(get_q(s))
                try:
                    pp = oracle_path(sentence=s, pi=pick_l2r_connected)
                    pp = [(list(p[0]), p[1], p[2]) for p in pp] 
                    of.write(json.dumps({"paths":pp, "sentence":s}) + "\n")
                    successes += 1
                except AssertionError:
                    failures += 1
    print("finished with {} successes and {} failures".format(successes, failures))

if __name__ == "__main__":

    train_fn = "preproc/training.jsonl"
    validation_fn = "preproc/validation.jsonl"


    make_paths(validation_fn)
    make_paths(train_fn)
