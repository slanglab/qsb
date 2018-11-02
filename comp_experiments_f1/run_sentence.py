import json
import csv
import argparse

from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy

def strip_tags(tokens):
    return [_ for _ in tokens if "SOS" not in _["word"] and "EOS" not in _["word"] and "OOV" not in _["word"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sentence", type=int)
    parser.add_argument("-model", type=str, default="nn")
    parser.add_argument("-archive_loc", type=str, default="tests/fixtures/891477193/model.tar.gz")
    parser.add_argument("-results_dir", type=str, default="comp_experiments_f1/output/")
    args = parser.parse_args()

    assert args.sentence is not None

    range_ = range(args.sentence * 100, args.sentence * 100 + 100)

    with open("preproc/lstm_validation.jsonl", "r") as inf:
        for vno, _ in enumerate(inf):
            if vno in range_:
                sentence = json.loads(_)
                sentence["tokens"] = strip_tags(sentence["tokens"])
                model = NeuralNetworkTransitionGreedy(args.archive_loc)
                orig_ix = sentence["original_ix"]
                y_true = [_ in sentence["compression_indexes"] for _ in orig_ix]
                out = model.predict(sentence)
                print(out)
                y_pred = out["y_pred"]
                ops = out["nops"]        
                f1 = f1_score(y_true=y_true, y_pred=y_pred)
                out_ = args.results_dir + "/{}-{}".format(vno,
                                                          args.model)
                with open(out_, "w") as of:
                    of.write(str(f1))
