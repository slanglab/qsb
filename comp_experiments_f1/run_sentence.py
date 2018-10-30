import json
import csv
import argparse

from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sentence", type=int)
    parser.add_argument("-model", type=str, default="nn")
    parser.add_argument("-archive_loc", type=str, default="tests/fixtures/633320059/model.tar.gz")
    parser.add_argument("-results_dir", type=str, default="comp_experiments_f1/output/")
    args = parser.parse_args()

    assert args.sentence is not None

    with open("preproc/validation.jsonl", "r") as inf:
        for vno, _ in enumerate(inf):
            if vno == args.sentence:
                sentence = json.loads(_)
                break

    if True: # args.model
        model = NeuralNetworkTransitionGreedy(args.archive_loc)
        orig_ix = [_["index"] for _ in sentence["compression_indexes"]]
        y_true = [_ in sentence["compression_indexes"] for _ in orig_ix]
        y_pred = model.predict(sentence)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        out_ = args.results_dir + "/{}-{}".format(args.sentence,
                                                  args.model)
        with open(out_, "w") as of:
            of.write(f1)