import json
import argparse

from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy


def strip_tags(tokens):
    return [_ for _ in tokens if "SOS" not in _["word"] and "EOS"
            not in _["word"] and "OOV" not in _["word"]]


def get_model(model):
    if model == "nn-greedy-query":
        return NeuralNetworkTransitionGreedy(config.archive_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-start", type=int)
    parser.add_argument("-config", type=int)
    args = parser.parse_args()

    with open(args.config, "r") as inf:
        config = json.load(inf)

    assert args.start is not None

    range_ = range(args.start * 100, args.start * 100 + 100)

    with open("preproc/lstm_validation.jsonl", "r") as inf:
        for vno, _ in enumerate(inf):
            if vno in range_:
                sentence = json.loads(_)
                sentence["tokens"] = strip_tags(sentence["tokens"])
                model = get_model(config["model"])
                orig_ix = sentence["original_ix"]
                y_true = [_ in sentence["compression_indexes"] for
                          _ in orig_ix]
                out = model.predict(sentence)
                y_pred = out["y_pred"]
                ops = out["nops"]
                f1 = f1_score(y_true=y_true, y_pred=y_pred)
                out_ = config.results_dir + "/{}-{}".format(vno,
                                                            config.model)
                config[sentence] = out_

    with open(out_, "w") as of:
        json.dump(config, of)
