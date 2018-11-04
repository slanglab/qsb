import json
import argparse

from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy
from comp_experiments_f1.algorithms import NeuralNetworkTransitionBFS


def strip_tags(tokens):
    return [_ for _ in tokens if "SOS" not in _["word"] and "EOS"
            not in _["word"] and "OOV" not in _["word"]]


def get_model(config):
    if config["model"] == "nn-greedy-query":
        query_focused = config["query"]
        print(query_focused)
        return NeuralNetworkTransitionGreedy(config["archive_loc"],
                                             query_focused)
    if config["model"] == "nn-transition-based":
        query_focused = config["query"]
        return NeuralNetworkTransitionBFS(config["archive_loc"],
                                          query_focused)
    assert "unknown" == "model"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-start", type=int)
    parser.add_argument("-config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as inf:
        config = json.load(inf)

    assert args.start is not None

    range_ = range(args.start * 100, args.start * 100 + 100)

    from tqdm import tqdm

    model = get_model(config)
    with open("preproc/lstm_validation_sentences_3way.jsonl", "r") as inf:
        for vno, _ in tqdm(enumerate(inf)):
            if vno in range_:
                sentence = json.loads(_)
                sentence["tokens"] = strip_tags(sentence["tokens"])
                orig_ix = sentence["original_ix"]
                y_true = [_ in sentence["compression_indexes"] for
                          _ in orig_ix]
                out = model.predict(sentence)
                y_pred = out["y_pred"]
                ops = out["nops"]
                f1 = f1_score(y_true=y_true, y_pred=y_pred)
                print(f1)
                config["sentence{}".format(vno)] = {'f1': f1, "nops": ops}

    out_ = config["results_dir"] + "/{}-{}".format(str(args.start),
                                                   config["model"])
    with open(out_, "w") as of:
        json.dump(config, of)
