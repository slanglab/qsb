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
parser.add_argument('-feature_config', type=str, default="bottom_up_clean/feature_config.json")
parser.add_argument('-random', dest='random', action='store_true', default=False)

args = parser.parse_args()

if __name__ == "__main__":

    with open(args.feature_config, "r") as inf:
        feature_config = json.load(inf)

    vocab = get_all_vocabs()

    if socket.gethostname() == "hobbes":
        from klm.query import LM, get_unigram_probs, slor
        lm = LM()
        unigram_log_probs_ = get_unigram_probs()

    clf, vectorizer, validationPreds = train_clf(training_paths=args.training_paths,
                                                 validation_paths=args.validation_paths,
                                                 feature_config=feature_config,
                                                 vocab=vocab)

    with open("bottom_up_clean/clf.p", "wb") as of:
        pickle.dump(clf, of)

    with open("bottom_up_clean/vectorizer.p", "wb") as of:
        pickle.dump(vectorizer, of)

    tot = 0
    slors = []

    print("[*] getting marginal")
    marginal = get_marginal(args.training_paths)

    totalNonTrees = 0
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
                                 decider=decider,
                                 vocab=vocab)
        compression = [_["word"] for _ in sentence["tokens"] if _["index"] in predicted]

        ### check if the sentence has any non trees?
        if socket.gethostname() == "hobbes":
            slors.append(slor(" ".join(compression), lm, unigram_log_probs_))
        else:
            slors.append(0)
        if has_forest(predicted, sentence):
            totalNonTrees += 1
        tot += get_f1(predicted, sentence)

    totalVal = sum(1 for i in open(args.validation_paths, "r"))

    with open("bottom_up_clean/results.csv", "a") as of:
        writer = csv.writer(of)
        out = [tot/totalVal, np.mean(slors), np.std(slors), decider.__name__]
        writer.writerow(out)
