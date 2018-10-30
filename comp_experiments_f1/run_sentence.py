import json
import csv
import argparse

from sklearn.metrics import f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=int)
    args = parser.parse_args()

    with open("preproc/validation.jsonl", "r") as inf:
        for vno, _ in enumerate(inf):
            if vno == args.sentence:
                sentence = json.loads(_)
                break

