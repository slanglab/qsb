from tqdm import tqdm
import numpy as np
import json
import random
import pickle
import argparse
import csv
import socket

# https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
from timeit import default_timer as timer

from sklearn.feature_extraction import DictVectorizer
from bottom_up_clean.all import make_decision_lr, make_decision_nn, runtime_path, pick_l2r_connected,make_decision_random, preproc, get_labels_and_features

if socket.gethostname() == "hobbes":
    from ilp2013.fillipova_altun import run_model

parser = argparse.ArgumentParser()
parser.add_argument('-path_to_set_to_evaluate', type=str, default="validation.paths")
parser.add_argument('-N', type=int, default=100000)
parser.add_argument('-ilp_snapshot', type=str, dest="ilp_snapshot", action='store')
args = parser.parse_args()


weights = "snapshots/" + args.ilp_snapshot


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


# you need to load everything into memory to sample fast
with open(args.path_to_set_to_evaluate, "r") as inf:
    S = []
    for ln in inf:
        ln = json.loads(ln)["sentence"]
        preproc(ln)
        S.append(ln)


def test_ILP(s):
    """Do compression"""
    run_model(s, r=s["r"], Q=s["q"], vectorizer=vectorizer, weights=weights)


def test_preproc(s):
    preproc(s)


def get_mean_var(f):
    all_ = []
    for i in tqdm(range(args.N)):
        s = random.sample(S, k=1)[0]
        start = timer()
        f(s)
        end = timer()
        all_.append({"s": s, "time": end - start})
    times = [_["time"] for _ in all_]
    return np.mean(times), np.std(times), all_


def test_additive(s):
    decider = make_decision_lr
    marginal = None

    runtime_path(s,
                 frontier_selector=pick_l2r_connected,
                 clf=clf,
                 vectorizer=vectorizer,
                 marginal=marginal,
                 decider=decider)


def test_additive_nn(s):
    decider = make_decision_nn
    marginal = None

    runtime_path(s,
                 frontier_selector=pick_l2r_connected,
                 clf=clf,
                 vectorizer=vectorizer,
                 marginal=marginal,
                 decider=decider)


def test_additive_at_random(s):
    decider = make_decision_random
    marginal = .3

    runtime_path(s,
                 frontier_selector=pick_l2r_connected,
                 clf=clf,
                 vectorizer=vectorizer,
                 marginal=marginal,
                 decider=decider)


def write_timing_results(all_, method):
    with open("bottom_up_clean/all_times.csv", "a") as of:
        for i in all_:
            ln = "{},{}\n".format(method, json.dumps(i))
            of.write(ln)


if __name__ == '__main__':
    with open("bottom_up_clean/timer.csv", "a") as of:
        writer = csv.writer(of)

        ## ILP
        with open(weights, "rb") as of:
            weights = pickle.load(of)

        #global clf and vectorizer b/c of timing problems 
        clf, vectorizer = get_clf_and_vectorizer()

        mean, var, all_ = get_mean_var(f=test_preproc)
        writer.writerow([mean, var, "test_preproc"])

        ## Full feature
        mean, var, all_ = get_mean_var(f=test_additive)
        writer.writerow([mean, var, "make_decision_lr"])
        write_timing_results(all_, "additive")

        # neural network
        mean, var, all_ = get_mean_var(f=test_additive_nn)
        writer.writerow([mean, var, "make_decision_nn"])
        write_timing_results(all_, "additive_nn")

        ### Only local vectorizer and classifier. Note reimport clf and vectorizer to only local version
        clf, vectorizer = get_clf_and_vectorizer(only_locals=True)
        mean,var, all_ = get_mean_var(f=test_additive)
        writer.writerow([mean, var, "only_locals"])
        write_timing_results(all_, "ablated")

        ## Random
        mean,var, all_ = get_mean_var(f=test_additive_at_random)
        writer.writerow([mean, var, "make_decision_random"]) 
        write_timing_results(all_, "random")

        ### now get the vectorizer for the ILP. again copying from FA learning, should be refactored at some point

        with open("preproc/training.paths",  "r") as inf:
            dataset = [_ for _ in inf]

        features, labels = get_labels_and_features(dataset, only_locals=True)
        vectorizer=DictVectorizer(sparse=True, sort=False)
        vectorizer.fit(features)

        for i in S:
            i["q"] = list(i['q'])

        mean, var, all_ = get_mean_var(f="test_ILP()", setup_="from __main__ import test_ILP")
        writer.writerow([mean, var, "ilp"])
        write_timing_results(all_, "ilp")
