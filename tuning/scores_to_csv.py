import json
import csv
import glob
scores = {}
best_epochs = {}

OUT = "/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/*/*metrics.json"

config = "/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/*/*config.json"

for fn in glob.glob(OUT):
    run = fn.split("/")[-2]
    print(fn)
    with open(fn, "r") as inf:
        dt = json.load(inf)
        scores[run] = dt["best_validation_accuracy"]
        best_epochs[run] = dt["best_epoch"]

configs = {}

for _ in glob.glob(config):
    #print(_)
    with open(_, "r") as inf:
        try:
            dt = json.load(inf)
            #print(_)
            #print(configs.keys())
            configs[_.split("/")[-2]] = dt
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
            print("***")
            print(_)
            print(scores[_])
    except KeyError:
        print(_)

with open("tuning/tuner.csv", "w") as of:
    csv_writer = csv.writer(of)
    csv_writer.writerows(out)
