
from bottom_up.all import *

if __name__ == "__main__":

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