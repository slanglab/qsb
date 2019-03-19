'''
python 3
'''
import json
import argparse
import pickle

from tqdm import tqdm
from sklearn.metrics import f1_score
from ilp2013.algorithms import FA2013Compressor
from ilp2013.algorithms imoprt FA2013CompressorStandard

from klm.query import LM
from klm.query import get_unigram_probs
from klm.query import slor


UNIGRAMS = get_unigram_probs()

LANGUAGE_MODEL = LM()


def get_model(config):
    print(config["algorithm"])

    if config['algorithm'][0:11] == "vanilla-ilp":
        with open(config["weights"], "rb") as of:
            weights = pickle.load(of)
        return FA2013CompressorStandard(weights=weights)

    if config["algorithm"][0:3] == "ilp":
        with open(config["weights"], "rb") as of:
            weights = pickle.load(of)
        return FA2013Compressor(weights=weights)

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
    compression = [o["word"] for ono, o in enumerate(sentence['tokens'])
                   if y_pred[ono]]
    compression = " ".join(compression)# SLOR implementation trained on stripped punct b/c models dont include punct b.c not in training data 

    lm_score = slor(sequence=compression,
                    lm=LANGUAGE_MODEL,
                    unigram_log_probs_=UNIGRAMS)

    config["sentence{}".format(vno)] = {'f1': f1,
                                        "lm": lm_score,
                                        "nops": ops,
                                        "prunes": prunes,
                                        "y_pred": y_pred,
                                        "y_true": y_true}


if __name__ == "__main__":

    model = get_model("ilp")

    fn = "preproc/validation.jsonl"

    config = {"algorithm": "ilp", "weights": "snapshots/1"}

    with open(fn, "r") as inf:
        no_compression = 0
        for vno, sent_ in tqdm(enumerate(inf)):
            do_sentence(sent_, no_compression, config)
