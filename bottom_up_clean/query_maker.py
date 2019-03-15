from collections import defaultdict
import numpy as np

import random


def draw_query_len():

    '''
    draw a query length

    distribution is based on "Real life, real users, and real needs: a study and analysis of user queries on the web"

    see table 6
    https://faculty.ist.psu.edu/jjansen/academic/pubs/ipm98/ipm98.pdf
    '''
    empirical = {6:.01, 5:.04, 4:.07, 3:.18, 2:.31, 1:.31, 0:0.08}   # all other query lengths

    items = list(empirical.items())
    items.sort(key=lambda x:x[0])
    p = [_[1] for _ in items]

    ou = 0
    while ou == 0:
        ou = np.where(np.random.multinomial(1, p, (1,))[0] > 0)[0]
        ou = ou[0]
    return ou


def draw_pos():

    empirical = {"proper-noun": .402, "noun": .309,
                 "adjective": .071, "other": .217}

    items = list(empirical.items())
    items.sort(key=lambda x:x[1], reverse=True)
    p = [_[1] for _ in items]

    pos = "other"
    while pos == "other":
        ou = np.where(np.random.multinomial(1, p, (1,))[0] > 0)[0]
        ou = ou[0]
        pos, probability = items[ou]
    return pos


def get_q(sentence):

    coursener = defaultdict(lambda: "other")
    coursener["NN"]="noun"
    coursener["NNS"]="noun"

    coursener["NNP"]="proper-noun"
    coursener["NNPS"]="proper-noun"# proper-noun (40.2%)

    coursener["JJ"]="adjective"#599 (7.1%)
    coursener["JJR"]="adjective"
    coursener["JJS"]="adjective"

    q = set()

    querylen = draw_query_len()

    cix = sentence["compression_indexes"]

    breaker = 0

    while(len(q) < querylen) and breaker < 10000:

        pos = draw_pos()

        qtok = [_ for _ in sentence["tokens"] if coursener[_["pos"]] == pos
                and _["index"] in cix and _["index"] not in q]

        random.shuffle(qtok)

        try:
            draw = qtok[0]
            q.add(draw["index"])
        except IndexError:
            pass
        breaker += 1

    assert len(q) > 0
    return q
