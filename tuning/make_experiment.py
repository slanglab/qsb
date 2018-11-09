from __future__ import division
import json
import random
import sys
import os
import glob

experiment = sys.argv[1]


# best configs one random run
#dropout,hidden_size,dropout_ff,weight_decay,lr,score
#0.025670178015253453,596,0.47406985591567374,0.00045421988121447727,0.001873627420156746,0.887912965365805 

for fn in glob.glob("/mnt/nfs/scratch1/ahandler/experiments/qsr/*json"):
    os.remove(fn)

def make_rando():

    with open(experiment, "r") as inf:
        dt = json.load(inf)

    dt['model']["abstract_encoder"]["hidden_size"] = random.randint(100,1000)
    dt['model']["abstract_encoder"]['dropout'] = random.uniform(0, .1)
    dt["model"]['classifier_feedforward_i']["dropout"] = random.uniform(.2, .7)
    dt["model"]['classifier_feedforward_i']["input_dim"] = dt['model']["abstract_encoder"]["hidden_size"] * 2


    dt["model"]['classifier_feedforward_p']["dropout"] = dt["model"]['classifier_feedforward_i']["dropout"]
    dt["model"]['classifier_feedforward_p']["input_dim"] = dt['model']["classifier_feedforward_i"]["input_dim"]

    x = random.uniform(2, 5)

    dt['trainer']['optimizer']["lr"] = 10 ** -x * random.uniform(1,10)

    x = random.uniform(3, 5)

    dt['trainer']['optimizer']['weight_decay'] = 10 ** -x * random.uniform(1, 10)

    import uuid

    uuid_ex = str(uuid.uuid1())[0:8]

    with open("/mnt/nfs/scratch1/ahandler/experiments/qsr/{}.json".format(uuid_ex), "w") as of:
        json.dump(dt, of)

for i in range(100):
    make_rando()
