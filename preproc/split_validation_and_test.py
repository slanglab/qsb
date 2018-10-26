'''
Make a validation and training set from the processed data
'''
import random
import json
import string
import glob
random.seed(1)


validation_size = 25000

def load_dataset():
    sources = []
    for source in glob.glob("sentence-compression/data/*sent-comp*source"):
        with open(source, "r") as inf:
            for ln in inf:
                ln = json.loads(ln)
                sources.append(ln)
    return sources


if __name__ == "__main__":
    data = load_dataset()
    random.shuffle(data)
    print "[*] dataset loaded"

    # this is for validation. note: no tree transform.
    # this will go to human experiments
    # the prune based models do not use transforms so the transform happens there
    with open("preproc/validation.jsonl", "w") as of:
        dt = "\n".join([json.dumps(_) for _ in data[-validation_size:]])
        of.write(dt)

    print "[*] dumped validation examples"
    # this is to train the ILP from F & A
    with open("preproc/training.jsonl", "w") as of:
        dt = [json.dumps(_) for _ in data[0:-validation_size]]
        print len(dt)
        of.write("\n".join(dt)) 
