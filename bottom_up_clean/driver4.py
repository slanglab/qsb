import json
import argparse

from bottom_up_clean.all import train_clf, runtime_path_wild_frontier, get_f1, pick_l2r_connected

parser = argparse.ArgumentParser()
parser.add_argument('-validation_paths', type=str, default="validation.paths")
parser.add_argument('-training_paths', type=str, default="training.paths")

args = parser.parse_args()

if __name__ == "__main__":
    clf, vectorizer = train_clf(training_paths=args.training_paths,
                                validation_paths=args.validation_paths)

    tot = 0
    for pno, paths in enumerate(open(args.validation_paths, "r")):
        paths = json.loads(paths)
        sentence = paths["sentence"]
        predicted = runtime_path_wild_frontier(sentence,
                                               frontier_selector=pick_l2r_connected,
                                               clf=clf,
                                               vectorizer=vectorizer)
        f1s = get_f1(predicted, sentence)
        tot += f1s
    print(tot/(pno + 1))
