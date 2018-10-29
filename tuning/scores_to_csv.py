import json
import csv
import glob
scores = {}
best_epochs = {}
for fn in glob.glob("experiments/*/*metrics.json"):
    run = fn.split("/")[1]
    with open(fn, "r") as inf:
        scores[run] = json.load(inf)["best_validation_accuracy"]
        best_epochs[run] = json.load(inf)["best_epoch"]

configs = {}

for _ in glob.glob("experiments/*/config.json"):
    with open(_, "r") as inf:
        try:
            dt = json.load(inf)
            configs[_.split("/")[1]] = dt
        except:
            pass


out = []

out.append(["dropout", "hidden_size", "dropout_ff", "weight_decay", "lr", "score"])

for _ in scores:
    stats = []
    try:
        cf = configs[_] 
        stats.append(cf["model"]["abstract_encoder"]["dropout"])
        stats.append(cf["model"]["abstract_encoder"]["hidden_size"])
        stats.append(cf["model"]["classifier_feedforward"]["dropout"]) 
        stats.append(cf["trainer"]["optimizer"]["weight_decay"])
        stats.append(cf["trainer"]["optimizer"]["lr"])
        stats.append(scores[_])
        if not type(stats[2]) == list:
            out.append(stats)
    except KeyError:
        print(_)

with open("tuning/tuner.csv", "w") as of:
    csv_writer = csv.writer(of)
    csv_writer.writerows(out)
