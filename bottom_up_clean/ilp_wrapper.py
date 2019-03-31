'''
wrapper for doing some measurements w/ the ILP
'''
import json
import argparse
import pickle
import copy
import numpy as np
from bottom_up_clean.all import get_labels_and_features
from sklearn.feature_extraction import DictVectorizer
from bottom_up_clean.all import preproc
from sklearn.metrics import f1_score
from ilp2013.fillipova_altun import run_model
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform
from ilp2013.fillipova_altun_supporting_code import get_pred_y
from klm.query import LM
from klm.query import get_unigram_probs
from klm.query import slor


class FA2013Compressor:

    '''
    This implements a query query_focused compression w/ F and A
    '''

    def __init__(self, weights, vectorizer):
        self.weights = weights
        self.vectorizer = vectorizer

    def predict(self, original_s, r, Q=[]):
        '''
        run the ILP.

        if Q = [] and r=len(s["headline"]) it is "normal" mode
        '''
        r = int(original_s["r"])

        original_indexes = [_["index"] for _ in original_s["tokens"]]

        transformed_s = filippova_tree_transform(copy.deepcopy(original_s))

        output = run_model(transformed_s,
                           weights=self.weights,
                           vectorizer=self.vectorizer,
                           Q=Q,
                           r=r)

        # note: not necessarily a unique set b/c multi edges possible to same
        # vertex after the transform. 
        predicted_compression = list(set([o['dependent'] for o in output["get_Xs"]]))
        y_pred = get_pred_y(predicted_compression=predicted_compression,
                            original_indexes=original_indexes)

        assert all([i in predicted_compression for i in original_s["q"]])
        assert len(output["compressed"]) <= original_s["r"]
        return {"y_pred": y_pred,
                "compression": output["compressed"],
                }


def do_sentence(sentence, no_compression, config, model, vno):
    
    orig_ix = [_["index"] for _ in sentence["tokens"]] 
    y_true = [_ in sentence["compression_indexes"] for
              _ in orig_ix]

    if config["algorithm"] == "vanilla-ilp":
        r = len(sentence["headline"])
        Q = []
    else:
        r = sentence["r"]
        Q = sentence["q"]

    out = model.predict(sentence, r=r, Q=Q)

    y_pred = out["y_pred"]
    if out["y_pred"] == "could not find a compression":
        f1 = 0.0
        no_compression += 1
        print("bad")
    else:
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
    assert f1 <= 1 and f1 >= 0
    compression = [o["word"] for ono, o in enumerate(sentence['tokens'])
                   if y_pred[ono] == 1]
    compression = " ".join(compression)# SLOR implementation trained on stripped punct b/c models dont include punct b.c not in training data 

    slor_score = slor(sequence=compression,
                     lm=LANGUAGE_MODEL,
                     unigram_log_probs_=UNIGRAMS)

    return {'f1': f1,
            "slor": slor_score,
            "y_pred": y_pred,
            "y_true": y_true}


def run_fn(config, fn, early_stop=None):

    with open(config["weights"], "rb") as of:
        weights = pickle.load(of)
    model = FA2013Compressor(weights=weights, vectorizer=config["vectorizer"])
    all_ = []
    with open(fn, "r") as inf:
        no_compression = 0
        for vno, sent_ in enumerate(inf):
            try:
                sent_ = json.loads(sent_)
                preproc(sent_)
                all_.append(do_sentence(sent_, no_compression, config, model, vno))
            except UnicodeError:
                print("Error")
            if early_stop is not None and  vno > early_stop: 
                break
    return all_


def assess_convergence():
    '''assess if the ILP has converged'''

    fn = "preproc/validation.jsonl"

    ## assess convergence

    for i in range(1, 6):

        config = {"vectorizer": vectorizer, "algorithm": "vanilla-ilp", "weights": "snapshots/{}".format(i)}

        jsonl = run_fn(config, fn, early_stop=1000)

        print("snapshot {}".format(i))
        print(np.mean([_["f1"] for _ in jsonl]))


if __name__ == "__main__":

    UNIGRAMS = get_unigram_probs()

    LANGUAGE_MODEL = LM()

    parser = argparse.ArgumentParser()
    parser.add_argument('-do_jsonl', type=str, default="validation.jsonl")
    parser.add_argument('-assess_convergence', dest='assess_convergence', action='store_true', default=False)
    parser.add_argument('-ilp_snapshot', type=str, dest="ilp_snapshot", action='store')
    args = parser.parse_args()

    # copying code from altun_learning here. 
    # should be refactored at some point but low priority
    with open("preproc/training.paths",  "r") as inf:
        dataset = [_ for _ in inf]

    features, labels = get_labels_and_features(dataset, only_locals=True)
    vectorizer=DictVectorizer(sparse=True, sort=False)
    vectorizer.fit(features) 

    if args.assess_convergence:
        assess_convergence()

    # algorithm, vanilla-ilp is regular
    config = {"vectorizer": vectorizer, 
              "algorithm": "ilp",
              "weights": "snapshots/{}".format(i)}

    jsonl = run_fn(config, args.do_jsonl, qsr=True, early_stop=10000)
    with open("bottom_up_clean/ilp_results.jsonl", "w") as of:
        for i in jsonl:
            of.write(json.dumps(i) + "\n")
