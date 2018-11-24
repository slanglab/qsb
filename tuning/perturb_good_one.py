##
# Used for minor perturbations of known good settings
# e.g. python tuning/perturb_good_one.py tuning/simple_and_big.json
##

from __future__ import division
import json
import random
import sys
import os
import glob

experiment = sys.argv[1]

for fn in glob.glob("/mnt/nfs/scratch1/ahandler/experiments/qsr/*json"):
    os.remove(fn)


def make_rando():

    with open(experiment, "r") as inf:
        dt = json.load(inf)

    if random.uniform(0, 1) < .5:
        dt['trainer']['optimizer']["lr"] += dt['trainer']['optimizer']["lr"] * random.uniform(.01,.03)
    else:
        dt['trainer']['optimizer']["lr"] -= dt['trainer']['optimizer']["lr"] * random.uniform(.01,.03)

    if random.uniform(0, 1) < .5:
        dt['trainer']['optimizer']['weight_decay'] += dt['trainer']['optimizer']['weight_decay'] * random.uniform(.01,.03)
    else:
        dt['trainer']['optimizer']['weight_decay'] -= dt['trainer']['optimizer']['weight_decay'] * random.uniform(.01,.03)

    if random.uniform(0, 1) < .5:
        dt['iterator']['batch_size'] += int(dt['iterator']['batch_size'] * random.uniform(.01,.03))
    else:
        dt['iterator']['batch_size'] -= int(dt['iterator']['batch_size'] * random.uniform(.01,.03))

    import uuid

    uuid_ex = str(uuid.uuid1())[0:8]

    with open("/mnt/nfs/scratch1/ahandler/experiments/qsr/{}.json".format(uuid_ex), "w") as of:
        json.dump(dt, of)

    print(dt)

for i in range(10):
    make_rando()
