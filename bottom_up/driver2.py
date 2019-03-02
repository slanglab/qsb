from bottom_up.all import get_features_and_labels
from bottom_up.all import get_lr
from bottom_up.all import bottom_up_from_clf
from bottom_up.all import f1_experiment

import json
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-eval_file', type=str)
    parser.add_argument('-train_file', type=str)
    parser.add_argument('-cutoff', type=int, default=10000000000)

    args = parser.parse_args()

    dev = [json.loads(_) for _ in open(args.eval_file)]

    feats_and_labels = get_features_and_labels(args.train_file,
                                               cutoff=args.cutoff)

    clf, vectorizer = get_lr(feats_and_labels)

    score = f1_experiment(dev,
                          bottom_up_from_clf,
                          clf=clf,
                          v=vectorizer)

    print(score)
