
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
                try:
                    #s["q"] = list(get_q(s))
                    pp = oracle_path(sentence=s, pi=pick_l2r_connected)
                    of.write(json.dumps({"paths":pp, "sentence":s}) + "\n")
                    successes += 1
                except AssertionError:
                    failures += 1
    print("finished with {} successes and {} failures".format(successes, failures))

if __name__ == "__main__":

    train_fn = "preproc/training.jsonl"
    validation_fn = "preproc/validation.jsonl"
    test_fn = "preproc/test.jsonl"

    #make_paths(validation_fn)
    #make_paths(train_fn)
    make_paths(test_fn)
