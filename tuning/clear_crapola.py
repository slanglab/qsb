### allen nlp stores huge files, most for bad hyperparams
### you want to keep the good ones to figure out good settings
### this clears out the bad to save disk

import json

for experiment in glob.glob("/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/*"):
    with open(experiment+ "/metrics_epoch_1.json") as inf:
        dt = json.load(inf)
