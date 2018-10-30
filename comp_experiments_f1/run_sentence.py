import json
import csv
import argparse

from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sentence", type=int)
    parser.add_argument("-model", type=str)
    parser.add_argument("-archive_loc", type=str, default="tests/fixtures/633320059/model.tar.gz")
    args = parser.parse_args()

    with open("preproc/validation.jsonl", "r") as inf:
        for vno, _ in enumerate(inf):
            if vno == args.sentence:
                sentence = json.loads(_)
                break

    if True: # args.model
        model = NeuralNetworkTransitionGreedy(args.archive_loc)
        preds = model.predict(sentence)
