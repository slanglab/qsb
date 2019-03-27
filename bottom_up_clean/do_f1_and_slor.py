import csv
import json
import pickle
import argparse
import socket
import numpy as np

from ilp2013.fillipova_altun_supporting_code import get_all_vocabs
from tqdm import tqdm

from bottom_up_clean.all import train_clf, runtime_path, get_f1, pick_l2r_connected, has_forest, get_marginal, make_decision_lr, make_decision_random


parser = argparse.ArgumentParser()
parser.add_argument('-validation_paths', type=str, default="validation.paths")
parser.add_argument('-training_paths', type=str, default="training.paths")
parser.add_argument('-random', dest='random', action='store_true', default=False)
parser.add_argument('-skip_globals', dest='skip_globals', action='store_true', default=False, help="don't use global features")
parser.add_argument('-skip_training', dest='skip_training', action='store_true', default=False)

args = parser.parse_args()

def do_training(training_paths, validation_paths, skip_globals = False):
    clf, vectorizer, validationPreds = train_clf(training_paths=training_paths,
                                                 validation_paths=validation_paths,
                                                 skip_globals=skip_globals)

    with open("bottom_up_clean/clf.p", "wb") as of:
        pickle.dump(clf, of)

    with open("bottom_up_clean/vectorizer.p", "wb") as of:
        pickle.dump(vectorizer, of)

    return clf, vectorizer

def get_scores():
    if socket.gethostname() == "hobbes":
        slor_score = slor(" ".join(compression), lm, unigram_log_probs_)
    else:
        slor_score = 0
    f1_score = get_f1(predicted, sentence)
    return slor_score, f1_score


if __name__ == "__main__":

    if socket.gethostname() == "hobbes":
        from klm.query import LM, get_unigram_probs, slor
        lm = LM()
        unigram_log_probs_ = get_unigram_probs()

    if not args.skip_training:
        # writes the .p files
        clf, vectorizer = do_training(args.training_paths, args.validation_paths, args.skip_globals)
    else:
        with open("bottom_up_clean/clf.p", "rb") as of:
            clf = pickle.load(of)

        with open("bottom_up_clean/vectorizer.p", "rb") as of:
            vectorizer = pickle.load(of)

    if args.random:
        print("[*] getting marginal")
        marginal = get_marginal(args.training_paths)
    else:
        marginal = None

    out = []
    tot = 0
    slors = []

    for pno, paths in enumerate(tqdm(open(args.validation_paths, "r"))):
        paths = json.loads(paths)
        sentence = paths["sentence"]
        if args.random is True:
            decider=make_decision_random
        else:
            decider=make_decision_lr

        predicted = runtime_path(sentence,
                                 frontier_selector=pick_l2r_connected,
                                 clf=clf,
                                 vectorizer=vectorizer,
                                 marginal=marginal,
                                 decider=decider)
        compression = [_["word"] for _ in sentence["tokens"] if _["index"] in predicted]

        slor_score, f1_score = get_scores()

        slors.append(slor_score)

        tot += f1_score

        out.append({"f1": f1_score, "slor": slor_score, "method": "additive"})

    totalVal = sum(1 for i in open(args.validation_paths, "r"))

    if args.skip_globals:
        name = "skipglobals"
    else:
        name = decider.__name__

    with open("bottom_up_clean/results.csv", "a") as of:
        writer = csv.writer(of)
        results = [tot/totalVal, np.mean(slors), np.std(slors), name]
        writer.writerow(results)

    with open("bottom_up_clean/additive_results.jsonl", "w") as of:
        for i in out:
            of.write(json.dumps(i) + "\n")
