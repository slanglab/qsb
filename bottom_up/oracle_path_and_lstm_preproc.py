
from bottom_up.all import *

from tqdm import tqdm


import pickle

train_fn = "preproc/training.jsonl"

dep2symbol = get_UD2symbols()

paths_loc = "bottom_up/" + train_fn.split("/").pop().replace(".jsonl", ".paths")

with open(paths_loc, "w") as of:
    with open(train_fn, "r") as inf:
        for _ in tqdm(inf): 
            failures = 0
            d = json.loads(_)
            try:
                p =  oracle_path(sentence=d, pi=pick_at_random)
                of.write(json.dumps({"paths":p, "sentence":d}))
            except:
                failures += 1

            print(failures)
'''
    with open(paths_loc, "r") as inf:
        for 
        for ino, item_ in enumerate(all_paths):
            paths, s = item_.values()
            for p in paths:
                t, v, y = p
                instance = get_instance(original_s=s, v=v, y=y, t=t, dep2symbol=dep2symbol)
                print(instance)
'''


def make_instances(fn):
    paths_loc = "bottom_up/" + fn.split("/").pop().replace(".jsonl", ".paths")

    failures = 0
    successes = 0
    with open(paths_loc, "w") as of:
        with open(fn, "r") as inf:
            for _ in tqdm(inf): 
                d = json.loads(_)
                try:
                    pp =  oracle_path(sentence=d, pi=pick_at_random)
                    pp = [(list(p[0]), p[1], p[2]) for p in pp] 
                    of.write(json.dumps({"paths":pp, "sentence":d}) + "\n")
                    successes += 1
                except AssertionError:
                    failures += 1
    print("Wrote paths w/ {} failures and {} successes".format(failures, successes))
    with open(paths_loc, "r") as inf:
        with open(paths_loc + ".dataset.jsonl", "w") as of:
            for ino, item_ in tqdm(enumerate(inf)):
                item_ = json.loads(item_)
                paths, s = item_.values()
                for p in paths:
                    t, v, y = p
                    instance = get_instance(original_s=s, v=v, y=y, t=t, dep2symbol=dep2symbol)
                    of.write(json.dumps(instance) + "\n")
 


if __name__ == "__main__":

    import pickle

    train_fn = "preproc/training.jsonl"
    validation_fn = "preproc/validation.jsonl"

    dep2symbol = get_UD2symbols()
   
    make_instances(validation_fn)
    make_instances(train_fn)