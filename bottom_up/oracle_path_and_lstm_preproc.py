
from bottom_up.all import *

from tqdm import tqdm
import json
from bottom_up.all import pick_at_random
from bottom_up.all import oracle_path
import pickle


def make_paths(fn):
    paths_loc = "bottom_up/" + fn.split("/").pop().replace(".jsonl", ".paths")

    failures = 0
    successes = 0
    with open(paths_loc, "w") as of:
        with open(fn, "r") as inf:
            for _ in tqdm(inf):
                s = json.loads(_)
                try:
                    pp =  oracle_path(sentence=s, pi=pick_bfs)
                    pp = [(list(p[0]), p[1], p[2]) for p in pp] 
                    of.write(json.dumps({"paths":pp, "sentence":s}) + "\n")
                    successes += 1
                except AssertionError:
                    failures += 1


if __name__ == "__main__":

    import pickle

    train_fn = "preproc/training.jsonl"
    validation_fn = "preproc/validation.jsonl"

    dep2symbol = get_UD2symbols()
   
    make_paths(validation_fn)
    make_paths(train_fn)
