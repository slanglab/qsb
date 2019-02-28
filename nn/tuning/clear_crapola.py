### allen nlp stores huge files, most for bad hyperparams
### you want to keep the good ones to figure out good settings
### this clears out the bad to save disk

import json
import glob

print(sum(1 for i in glob.glob("/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/*")))

crapola = []

for experiment in glob.glob("/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/*"):
    try:
        with open(experiment+ "/metrics_epoch_1.json") as inf:
            dt = json.load(inf)
            if dt["best_validation_accuracy"] < .85:
                crapola.append(experiment)
    except FileNotFoundError:
        pass

import shutil
for c in crapola:
    shutil.rmtree(c)

print("removed {} failed experiments".format(len(crapola)))
