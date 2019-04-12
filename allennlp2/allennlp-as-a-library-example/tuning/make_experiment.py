
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

    inputd = [300, 100]
    random.shuffle(inputd)
    inputd = inputd[0]

    nonlinearity = ["sigmoid", "relu", "tanh", "linear"]
    random.shuffle(nonlinearity)

    with open(experiment, "r") as inf:
        dt = json.load(inf)

    fn = "/home/ahandler/qsr/glove.6B.{}d.txt.gz".format(inputd)
    dt['model']['text_field_embedder']["token_embedders"]['tokens']['embedding_dim'] = inputd

    dt["model"]["title_encoder"]["embedding_dim"] = inputd
    dt["model"]["abstract_encoder"]["embedding_dim"] = inputd

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

    for component in ['classifier_feedforward']:
        dt["model"][component]["dropout"] = dropouts
        dt["model"][component]["input_dim"] = inputd 
        dt["model"][component]["activations"] = activations_
        dt["model"][component]["num_layers"] = classification_layers
        dt["model"][component]["hidden_dims"] = sizes

    x = random.uniform(3, 5)

    dt['trainer']['optimizer']["lr"] = 10 ** -x * random.uniform(1, 10)

    x = random.uniform(5, 8)

    dt['trainer']['optimizer']['weight_decay'] = 10 ** -x * random.uniform(1, 10)

    dt['iterator']['batch_size'] = int(random.uniform(50, 150))

    import uuid

    uuid_ex = str(uuid.uuid1())[0:8]

    with open("/mnt/nfs/scratch1/ahandler/experiments/qsr/{}.json".format(uuid_ex), "w") as of:
        json.dump(dt, of)

    print(dt)

for i in range(100):
    make_rando()
