from tqdm import tqdm
import numpy as np
import json
import random
import pickle
import argparse
import timeit
import csv
import socket

from bottom_up_clean.all import make_decision_lr, runtime_path, pick_l2r_connected,make_decision_random

if socket.gethostname() == "hobbes":
    from ilp2013.fillipova_altun import run_model
    from ilp2013.fillipova_altun_supporting_code import get_all_vocabs
    vocabs = get_all_vocabs()

parser = argparse.ArgumentParser()
parser.add_argument('-path_to_set_to_evaluate', type=str, default="validation.paths")
parser.add_argument('-N', type=int, default=100000)
args = parser.parse_args()

S = []

weights = "snapshots/1"


def get_clf_and_vectorizer(only_locals=False):

    if only_locals:
        clf = "bottom_up_clean/clf_only_locals.p"
        vectorizer = "bottom_up_clean/vectorizer_only_locals.p"
    else:
        clf = "bottom_up_clean/clf.p"
        vectorizer = "bottom_up_clean/vectorizer.p"

    with open(clf, "rb") as of:
        clf = pickle.load(of)

    with open(vectorizer, "rb") as of:
        vectorizer = pickle.load(of)
    return clf, vectorizer


with open(args.path_to_set_to_evaluate, "r") as inf:
    for ln in inf:
        ln = json.loads(ln)
        S.append(ln)


def test_ILP():
    """Do compression"""
    s = random.sample(S, k=1)[0]["sentence"]
    run_model(s, r=s["r"], Q=s["q"], vocab=vocabs, weights=weights) 

def get_mean_var(f='test_ILP()',setup_='from __main__ import test_ILP'):
    all_ = []
    for i in tqdm(range(args.N)):
        a = timeit.timeit(f, setup=setup_, number=1)
        all_.append(a)
    return np.mean(all_), np.std(all_)
    

def test_additive():
    decider=make_decision_lr
    marginal = None
    sentence = random.sample(S, k=1)[0]["sentence"]

    runtime_path(sentence,
                 frontier_selector=pick_l2r_connected,
                 clf=clf,
                 vectorizer=vectorizer,
                 marginal=marginal,
                 decider=decider)


def test_additive_at_random():
    decider=make_decision_random
    marginal = .3
    sentence = random.sample(S, k=1)[0]["sentence"]

    runtime_path(sentence,
                 frontier_selector=pick_l2r_connected,
                 clf=clf,
                 vectorizer=vectorizer,
                 marginal=marginal,
                 decider=decider)


if __name__ == '__main__':
    with open("bottom_up_clean/timer.csv", "a") as of:
        writer = csv.writer(of)

        #global clf and vectorizer b/c of timing problems 
        clf, vectorizer = get_clf_and_vectorizer()

        ## Full feature
        mean,var = get_mean_var(f="test_additive()", setup_="from __main__ import test_additive")
        writer.writerow([mean, var, "make_decision_lr"]) 

        ### Only local vectorizer and classifier. Note reimport clf and vectorizer to only local version
        clf, vectorizer = get_clf_and_vectorizer(only_locals=True)
        mean,var = get_mean_var(f="test_additive()", setup_="from __main__ import test_additive")
        writer.writerow([mean, var, "make_decision_only_locals"])

        ## Random
        mean,var = get_mean_var(f="test_additive_at_random()", setup_="from __main__ import test_additive_at_random")
        writer.writerow([mean, var, "make_decision_random"]) 

        ## ILP
        with open(weights, "rb") as of:
            weights = pickle.load(of)

        mean,var = get_mean_var(f="test_ILP()", setup_="from __main__ import test_ILP")
        writer.writerow([mean, var, "ilp"])
 
       

       
