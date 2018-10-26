import glob
import json
import itertools
import gzip
import sys
import cPickle as pickle
from collections import defaultdict


search_dir = sys.argv[1] # sentence-compression/data for production, tests/ for test

vocabs = [json.load(open(_, "r")) for _ in glob.glob(search_dir + "/*.vocabs")]

all_vocabs = vocabs[0].keys()

all_vs = defaultdict()

print all_vocabs

for v in all_vocabs:
    
    l = list(itertools.chain(*[_[v] for _ in vocabs]))
    with open(search_dir + "/" + v, "w") as of:
        of.write("\n".join([_ for _ in set(l)]))
        all_vs[v] = [_ for _ in set(l)]


with gzip.open(search_dir + "/all_vocabs.json.p", "w") as of:
    pickle.dump(dict(all_vs), of, -1)
