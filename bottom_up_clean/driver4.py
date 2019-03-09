import json
import pickle
import argparse

from bottom_up_clean.all import train_clf, runtime_path, get_f1, pick_l2r_connected

with open("empirical.csv", "r") as inf:
    lns = [_.replace("\n", "").split(",") for _ in inf]
    lns = [(_[0], float(_[1])) for _ in lns]


other = 100 - sum([_[1] for _ in lns])
lns.append((0, other))

empirical_length_probability = [_[1]/100 for _ in lns]


POS = {
"proper-noun":["NNP", "NNPS"], # proper-noun (40.2%)
"noun":["NN", "NNP"], #2601 (30.9%)
"adjective":["JJ", "JJR", "JJS"], #599 (7.1%)
"preposition": ["IN"] # (3.7%)
}

parser = argparse.ArgumentParser()
parser.add_argument('-validation_paths', type=str, default="validation.paths")
parser.add_argument('-training_paths', type=str, default="training.paths")

args = parser.parse_args()


if __name__ == "__main__":
    clf, vectorizer, validationPreds = train_clf(training_paths=args.training_paths,
                                                 validation_paths=args.validation_paths)

    tot = 0

    totalNonTrees = 0
    for pno, paths in enumerate(open(args.validation_paths, "r")):
        paths = json.loads(paths)
        sentence = paths["sentence"]
        predicted = runtime_path(sentence,
                                 frontier_selector=pick_l2r_connected,
                                 clf=clf,
                                 vectorizer=vectorizer)

        ### check if the sentence has any non trees?
        hit_sentence = False
        for p in predicted:
            gov = [_['governor'] for _ in sentence["basicDependencies"] if _["dependent"] == p][0]
            if gov not in predicted | {0}:
                hit_sentence = True
        if hit_sentence:
            totalNonTrees += 1

        f1s = get_f1(predicted, sentence)
        tot += f1s
    totalVal = sum(1 for i in open(args.validation_paths, "r"))
    print("F1={}".format(tot/(totalVal)))
    print("Pct. forest={}".format(totalNonTrees / totalVal))

