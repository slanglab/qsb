'''
read in a .arpa file and make a dictionary of all ungirams to their probability

this should run in the klm directory
'''

import json
from tqdm import tqdm
from collections import defaultdict

collector = defaultdict()

FN = 'gigaword.arpa.unigram'

with open("../cache/vocab.json") as inf:
    vocab = set(json.load(inf))

with open(FN, "r") as inf:
    for ln in tqdm(inf):
        ln = ln[0:-1]
        if len(ln.split("\t")) == 2:
            k,v = ln.split("\t")
            if "\n" not in v and v in vocab:
                collector[v] = float(k)

with open("unigram.json", "w") as of:
    json.dump(collector, of)
