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

tok_indexers = {"tokens": {"type": "single_id"},
                "elmo": {"type": "elmo_characters"}}

elmo_vectors = {"type": "elmo_token_embedder",
                "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                "do_layer_norm": False,
                "dropout": 0.5
                }


def make_rando():

    inputd = [300, 100]
    random.shuffle(inputd)
    inputd = inputd[0]

    nonlinearity = ["sigmoid", "relu", "tanh", "linear"]
    random.shuffle(nonlinearity)

    elmo = [False]#, True]
    random.shuffle(elmo)

    #nonlinearity = nonlinearity[0]

    with open(experiment, "r") as inf:
        dt = json.load(inf)

    if elmo:
        dt["dataset_reader"]["token_indexers"] = tok_indexers
        dt["dataset_reader"]["text_field_embedder"] = elmo_vectors

    fn = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.{}d.txt.gz".format(inputd)
    print(dt["model"]["text_field_embedder"]["tokens"])
    dt['model']['text_field_embedder']["tokens"]['pretrained_file'] = fn
    dt['model']['text_field_embedder']['tokens']['embedding_dim'] = inputd

    if elmo:
        dt["model"]["abstract_encoder"]["input_size"] = inputd + 1024
    else:
        dt["model"]["abstract_encoder"]["input_size"] = inputd

    dt['model']["abstract_encoder"]["hidden_size"] = random.randint(100, 1000)
    dt['model']["abstract_encoder"]['dropout'] = random.uniform(0, .1)

    classification_layers = range(1, 4)
    random.shuffle(classification_layers)
    classification_layers = classification_layers[0]

    max_ = 100
    activations_ = []
    sizes = []
    dropouts = []
    for i in range(classification_layers):
        activations_.append(random.choice(nonlinearity))
        sizes.append(random.choice(range(max_)))
        max_ = max_ / 2
        dropouts.append(random.uniform(.1, .7))

    for component in ['classifier_feedforward_i', 'classifier_feedforward_p']:
        dt["model"][component]["dropout"] = dropouts
        dt["model"][component]["input_dim"] = dt['model']["abstract_encoder"]["hidden_size"] * 2
        dt["model"][component]["activations"] = activations_
        dt["model"][component]["num_layers"] = classification_layers
        dt["model"][component]["hidden_dims"] = sizes

    layers = [1, 2, 3]
    random.shuffle(layers)
    num_layers = layers[0]
    dt["model"]["abstract_encoder"]["num_layers"] = num_layers

    x = random.uniform(3, 6)

    dt['trainer']['optimizer']["lr"] = 10 ** -x * random.uniform(1, 10)

    x = random.uniform(3, 10)

    dt['trainer']['optimizer']['weight_decay'] = 10 ** -x * random.uniform(1, 10)

    dt['iterator']['batch_size'] = int(random.uniform(50, 150))

    import uuid

    uuid_ex = str(uuid.uuid1())[0:8]

    with open("/mnt/nfs/scratch1/ahandler/experiments/qsr/{}.json".format(uuid_ex), "w") as of:
        json.dump(dt, of)

    print(dt)

for i in range(100):
    make_rando()
