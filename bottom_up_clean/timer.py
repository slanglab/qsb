
import json
import random
import pickle
from ilp2013.fillipova_altun import run_model
from ilp2013.fillipova_altun_supporting_code import get_all_vocabs

fn = "preproc/lstm_validation_sentences_3way.jsonl"

S = []

weights = "snapshots/8"

with open(weights, "rb") as of:
    weights = pickle.load(of)

vocabs = get_all_vocabs()

with open(fn, "r") as inf:
    for ln in inf:
        ln = json.loads(ln)
        S.append(ln)

def test():
    """Stupid test function"""
    s = random.sample(S, k=1)[0]
    run_model(s, r=s["r"], Q=s["q"], vocab=vocabs, weights=weights) 

if __name__ == '__main__':
    import timeit
    print(timeit.repeat("test()", repeat=3, number=100, setup="from __main__ import test"))
