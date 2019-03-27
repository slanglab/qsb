'''
python 3
'''
import json
import argparse
import pickle

from tqdm import tqdm
from sklearn.metrics import f1_score
from ilp2013.algorithms import FA2013Compressor
from ilp2013.algorithms import FA2013CompressorStandard
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform

from klm.query import LM
from klm.query import get_unigram_probs
from klm.query import slor


UNIGRAMS = get_unigram_probs()

LANGUAGE_MODEL = LM()



def get_model(config):
    print(config)
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


def do_sentence(_, no_compression, config, model, vno):
    sentence = json.loads(_)
    
    orig_ix = [_["index"] for _ in sentence["tokens"]] 
    y_true = [_ in sentence["compression_indexes"] for
              _ in orig_ix]
    out = model.predict(sentence)
    y_pred = out["y_pred"]
    if out["y_pred"] == "could not find a compression":
        f1 = 0.0
        no_compression += 1
        with open("/tmp/{}".format(vno), "wb") as of:
            pickle.dump(sentence, of)
    else:
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
    assert f1 <= 1 and f1 >= 0
    compression = [o["word"] for ono, o in enumerate(sentence['tokens'])
                   if y_pred[ono]]
    compression = " ".join(compression)# SLOR implementation trained on stripped punct b/c models dont include punct b.c not in training data 

    slor_score = slor(sequence=compression,
                     lm=LANGUAGE_MODEL,
                     unigram_log_probs_=UNIGRAMS)

    config["sentence{}".format(vno)] = {'f1': f1,
                                        "slor": slor_score,
                                        "y_pred": y_pred,
                                        "y_true": y_true}


def get_metric_from_config(config, metric):
    f1s = 0
    total = 0
    for o in config:
        if "sentence" in o:
            f1s += config[o][metric]
            total += 1
    return f1s/total

def config2json(config):
    out = []
    for o in config:
        if "sentence" in o:
            out.append({"f1": config[o]["f1"], "slor": config[o]["slor"]})
    return out 

def run_fn(config, fn, early_stop=None):

    model = get_model(config)

    with open(fn, "r") as inf:
        no_compression = 0
        for vno, sent_ in enumerate(inf):
            try:
                do_sentence(sent_, no_compression, config, model, vno)
            except:
                print("Error")
            if early_stop is not None and  vno > early_stop: 
                break
    return config


def assess_convergence():
    '''assess if the ILP has converged'''

    fn = "preproc/validation.jsonl"

    ## assess convergence

    for i in range(1, 6):

        config = {"algorithm": "vanilla-ilp", "weights": "snapshots/{}".format(i)}

        config = run_fn(config, fn, early_stop=1000)

        print("snapshot {}".format(i))
        print(get_F1_from_config(config))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-do_jsonl', type=str, default="validation.jsonl")
    parser.add_argument('-assess_convergence', dest='assess_convergence', action='store_true', default=False)

    args = parser.parse_args()

    if args.assess_convergence:
        assess_convergence()

    config = {"algorithm": "ilp", "weights": "snapshots/1"}

    model = get_model(config)

    run_fn(config, args.do_jsonl, early_stop=10000)
    print(get_metric_from_config(config, "f1"))
    print(get_metric_from_config(config, "slor"))
    with open("bottom_up_clean/ilp.json", "w") as of:
        for i in config2json(config):
            of.write(json.dumps(i) + "\n")
