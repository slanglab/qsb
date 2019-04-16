
from __future__ import division
import json
import random
import sys
import os
import glob
from random import uniform

experiment = sys.argv[1]


for fn in glob.glob("/mnt/nfs/scratch1/ahandler/experiments/qsr/*json"):
    os.remove(fn)

from random import randint

def make_rando():

    inputd = randint(50, 2000) 

    nonlinearity = ["sigmoid", "relu", "tanh", "linear"]
    random.shuffle(nonlinearity)

    with open(experiment, "r") as inf:
        dt = json.load(inf)

    dt['model']['text_field_embedder']["token_embedders"]['tokens']['embedding_dim'] = inputd

    dt["model"]["abstract_encoder"]["input_size"] = inputd
    dt["model"]["abstract_encoder"]["dropout"] = uniform(0,1)
    hidden_size = randint(15,200)
    dt['model']['hidden_size'] = hidden_size 
    classification_layers = list(range(1, 4))
    random.shuffle(classification_layers)
    classification_layers = classification_layers[0]

    max_ = 100
    activations_ = []
    sizes = []
    dropouts = []
    for i in range(classification_layers):
        activations_.append(random.choice(nonlinearity))
        sizes.append(random.choice(range(int(max_))))
        max_ = int(max_ / 2)
        dropouts.append(random.uniform(.1, .7))

    sizes[-1] = 2
    dt["model"]["classifier_feedforward"]["dropout"] = hidden_size * 2
    for component in ['classifier_feedforward']:
        dt["model"][component]["dropout"] = dropouts
        #dt["model"][component]["input_dim"] = inputd 
        dt["model"][component]["activations"] = activations_
        dt["model"][component]["num_layers"] = classification_layers
        dt["model"][component]["hidden_dims"] = sizes

    x = random.uniform(1,4)

    dt['trainer']['optimizer']["lr"] = 10 ** -x * random.uniform(1, 10)

    x = random.uniform(1, 10)

    dt['trainer']['optimizer']['weight_decay'] = 10 ** -x * random.uniform(1, 10)

    dt['iterator']['batch_size'] = int(random.uniform(50, 150))

    import uuid

    uuid_ex = str(uuid.uuid1())[0:8]

    with open("/mnt/nfs/scratch1/ahandler/experiments/qsr/{}.json".format(uuid_ex), "w") as of:
        json.dump(dt, of)

    print(dt)

for i in range(25):
    make_rando()
