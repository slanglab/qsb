
import json
from ilp2013.fillipova_altun import run_model

fn = "preproc/lstm_validation_sentences_3way.jsonl"

S = []

with open(fn, "r") as inf:
    for ln in inf:
        ln = json.loads(ln)
        S.append(ln)

def test():
    """Stupid test function"""
    L = [i for i in range(100)]

if __name__ == '__main__':
    import timeit
    print(timeit.repeat("test()", repeat=3, number=100, setup="from __main__ import test"))
