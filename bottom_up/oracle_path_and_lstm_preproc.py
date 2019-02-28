
from bottom_up.all import *

if __name__ == "__main__":

    import pickle
    dev = [json.loads(_) for _ in open("dev.jsonl")]

    paths = []
    
    failures = 0
    for d in dev:
        try:
            p =  oracle_path(sentence=d, pi=pick_at_random)
            paths.append(({"paths":p, "sentence":d}))
        except:
            failures += 1

    print(failures)
    
    with open("/tmp/paths.p", "wb") as of:
        pickle.dump(paths, of)

    with open("/tmp/paths.p", "rb") as of:
        all_paths = pickle.load(of)

    dep2symbol = get_UD2symbols()

    for ino, item_ in enumerate(all_paths):
        paths, s = item_.values()
        for p in paths:
            t, v, y = p
            instance = get_instance(original_s=s, v=v, y=y, t=t, dep2symbol=dep2symbol)
            print(instance)