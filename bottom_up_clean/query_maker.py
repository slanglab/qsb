from collections import defaultdict
import numpy as np

import random
from random import shuffle


def draw_query_len():

    '''
    draw a query length, 1 to 3 nouns

    https://faculty.ist.psu.edu/jjansen/academic/pubs/ipm98/ipm98.pdf
    '''
    ou = [1, 2, 3]
    shuffle(ou)
    return ou[0]


def get_q(sentence):

    coursener = defaultdict(lambda: "other")
    coursener["NN"] = "noun"
    coursener["NNS"] = "noun"

    coursener["NNP"] = "noun"
    coursener["NNPS"] = "noun" # proper-noun (40.2%)

    q = set()

    querylen = draw_query_len()

    cix = sentence["compression_indexes"]

    nouns = [_["index"] for _ in sentence["tokens"] if
             coursener[_["pos"]] == "noun" and _["index"] in cix]

    shuffle(nouns)

    assert len(q) > 0
    return set(nouns[0:querylen])
