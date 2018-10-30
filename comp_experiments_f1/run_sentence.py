import json
import csv
import argparse

from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sentence", type=int)
    parser.add_argument("-model", type=str)
    args = parser.parse_args()

    with open("preproc/validation.jsonl", "r") as inf:
        for vno, _ in enumerate(inf):
            if vno == args.sentence:
                sentence = json.loads(_)
                break

    if True: # args.model
        model = NeuralNetworkTransitionGreedy()
        preds = model.predict.predict(sentence)
