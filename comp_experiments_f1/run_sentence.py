'''
python 3
'''
import json
import argparse
import pickle

from tqdm import tqdm
from sklearn.metrics import f1_score
from comp_experiments_f1.algorithms import NeuralNetworkTransitionGreedy
from comp_experiments_f1.algorithms import NeuralNetworkTransitionBFS
from comp_experiments_f1.algorithms import NeuralNetworkPredictThenPrune
from comp_experiments_f1.algorithms import BaselineCompressor
from comp_experiments_f1.algorithms import FA2013Compressor
from comp_experiments_f1.algorithms import FMCSearch


def strip_tags(tokens):
    return [_ for _ in tokens if "SOS" not in _["word"] and "EOS"
            not in _["word"] and "OOV" not in _["word"]]


def get_model(config):
    print(config["algorithm"])
    if config["algorithm"] == "nn-prune-greedy":
        query_focused = config["query"]
        model_name = config["model_name"]
        predictor_name = config["predictor_name"]
        print(query_focused)
        return NeuralNetworkTransitionGreedy(archive_loc=config["archive_loc"],
                                             query_focused=query_focused,
                                             predictor_name=predictor_name,
                                             model_name=model_name)

    if config["algorithm"] == "min-compression":
        return BaselineCompressor()

    if config["algorithm"] == "transition-based-compressor":
        query_focused = config["query"]
        model_name = config["model_name"]
        predictor_name = config["predictor_name"]
        return NeuralNetworkTransitionBFS(archive_loc=config["archive_loc"],
                                          model_name=model_name,
                                          predictor_name=predictor_name)

    if config["algorithm"] == "fmcsearch":
        query_focused = config["query"]
        model_name = config["model_name"]
        predictor_name = config["predictor_name"]
        return FMCSearch(archive_loc=config["archive_loc"], 
                         predictor_name=predictor_name,
                         model_name=model_name,
                         nsamples=config["nsamples"])

    if config["algorithm"] == "ilp":
        with open(config["weights"], "rb") as of:
            weights = pickle.load(of)
        return FA2013Compressor(weights=weights)

    if config['algorithm'] == "predict-then-prune":
        query_focused = config["query"]
        model_name = config["model_name"]
        predictor_name = config["predictor_name"]
        return NeuralNetworkPredictThenPrune(archive_loc=config["archive_loc"],
                                             query_focused=query_focused,
                                             predictor_name=predictor_name,
                                             model_name=model_name)

    assert "unknown" == "model"


def do_sentence(_, no_compression, config):
    sentence = json.loads(_)
    sentence["tokens"] = strip_tags(sentence["tokens"])
    orig_ix = sentence["original_ix"]
    y_true = [_ in sentence["compression_indexes"] for
              _ in orig_ix]
    out = model.predict(sentence)
    y_pred = out["y_pred"]
    ops = out["nops"]
    if "prunes" in out:
        prunes = out["prunes"]
    else:
        prunes = -999999
    if out["y_pred"] == "could not find a compression":
        f1 = 0.0
        no_compression += 1
        with open("/tmp/{}".format(vno), "wb") as of:
            pickle.dump(sentence, of)
    else:
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        if args.verbose:
            print("***")
            print(" ".join([o["word"] for o in sentence["tokens"]]))
            print(" ".join([o["word"] for ino, o in enumerate(sentence["tokens"])
                            if y_pred[ino]]))
    assert f1 <= 1 and f1 >= 0
    config["sentence{}".format(vno)] = {'f1': f1,
                                        "nops": ops,
                                        "prunes": prunes,
                                        "y_pred": y_pred,
                                        "y_true": y_true}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fast", action="store_true", default=False)
    parser.add_argument("-config", type=str)
    parser.add_argument("-verbose", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config, "r") as inf:
        config = json.load(inf)

    if args.fast:
        range_ = range(0, 100)
    else:
        range_ = range(0, 1000)

    model = get_model(config)
    with open("preproc/lstm_validation_sentences_3way.jsonl", "r") as inf:
        no_compression = 0
        for vno, _ in tqdm(enumerate(inf)):
            print(vno)
            if vno in range_:
                try:
                    do_sentence(_, no_compression, config) 
                except IndexError:
                    print("ERROR")

    fast = "fast" if args.fast else "full"
    archive = config["archive_loc"]
    out_ = config["results_dir"] + "/{}-{}-{}".format(str(fast), archive,
                                                   config["algorithm"])
    config["no_compression"] = no_compression 
    print(config.keys())
    with open(out_, "w") as of:
        json.dump(config, of)
