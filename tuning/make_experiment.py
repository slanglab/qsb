from __future__ import division
import json
import random


def make_rando():

    with open("nn/op_classifier.json", "r") as inf:
        dt = json.load(inf)

    dt['model']["abstract_encoder"]["hidden_size"] = random.randint(10, 2000)
    dt['model']["abstract_encoder"]['dropout'] = random.uniform(0, .5)
    dt["model"]['classifier_feedforward']["dropout"] = random.uniform(0, .5)
    dt["model"]['classifier_feedforward']["input_dim"] = dt['model']["abstract_encoder"]["hidden_size"] * 2

    x = random.uniform(1, 8)

    dt['trainer']['optimizer']["lr"] = 10 ** -x * random.uniform(1,10)

    x = random.uniform(1, 8)

    dt['trainer']['optimizer']['weight_decay'] = 10 ** -x * random.uniform(1, 10)

    import uuid

    uuid_ex = str(uuid.uuid1())[0:8]

    with open("/mnt/nfs/scratch1/ahandler/experiments/qsr/{}.json".format(uuid_ex), "w") as of:
        json.dump(dt, of)

for i in range(100):
    make_rando()
