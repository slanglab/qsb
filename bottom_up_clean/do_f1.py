import json
import pickle
import argparse
import numpy as np

from bottom_up_clean.all import train_clf, runtime_path, get_f1, pick_l2r_connected, has_forest, get_marginal, make_decision_lr, make_decision_random

from klm.query import slor, LM, get_unigram_probs

parser = argparse.ArgumentParser()
parser.add_argument('-validation_paths', type=str, default="validation.paths")
parser.add_argument('-training_paths', type=str, default="training.paths")
parser.add_argument('-random', dest='random', action='store_true', default=False)

args = parser.parse_args()

lm = LM()

unigram_log_probs_ = get_unigram_probs() 

if __name__ == "__main__":
    clf, vectorizer, validationPreds = train_clf(training_paths=args.training_paths,
                                                 validation_paths=args.validation_paths)

    with open("bottom_up_clean/clf.p", "wb") as of:
        pickle.dump(clf, of)

    with open("bottom_up_clean/vectorizer.p", "wb") as of:
        pickle.dump(vectorizer, of)

    tot = 0
    slors = []

    marginal = get_marginal(args.training_paths)

    totalNonTrees = 0
    for pno, paths in enumerate(open(args.validation_paths, "r")):
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
        slors.append(slor(" ".join(compression), lm=lm, unigram_log_probs_=unigram_log_probs_))
        ### check if the sentence has any non trees?
        if has_forest(predicted, sentence):
            totalNonTrees += 1

        tot += get_f1(predicted, sentence)
    totalVal = sum(1 for i in open(args.validation_paths, "r"))
    print("F1={}".format(tot/(totalVal)))
    print("Pct. forest={}".format(totalNonTrees / totalVal)) 
    print("slor mean/std={},{}".format(np.mean(slors), np.std(slors)))
