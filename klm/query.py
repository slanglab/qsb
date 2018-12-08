from __future__ import division

import kenlm
import json

LOC = "/home/ahandler/qsr/klm/fa.klm"
UNIGRAM_LOC = "/home/ahandler/qsr/klm/fa.unigrams.json"


def get_unigram_probs():
    with open(UNIGRAM_LOC, "r") as inf:
        unigram_log_probs = json.load(inf)
        unigram_log_probs = {k:float(v) for k,v in unigram_log_probs.items()}
        return unigram_log_probs


def slor(sequence, lm, unigram_log_probs_):
    # SLOR function from Jey Han Lau, Alexander Clark, and Shalom Lappin

    words = sequence.split(" ")
    p_u = sum(unigram_log_probs_[u] for u in words if u in unigram_log_probs_.keys())
    p_u += sum(unigram_log_probs_['<unk>'] for u in words if u not in unigram_log_probs_.keys())

    p_m = lm.score(sequence)

    len_s = len(words) + 0.0

    return (p_m - p_u)/len_s


class LM:

    def __init__(self):

        self.model = kenlm.LanguageModel(LOC)

    def score(self, str_):
        # str_ is a " "-delimited string, e.g. "I am a student"

        return self.model.score(str_)


if __name__ == "__main__":

    model = LM()

    ug = get_unigram_probs() 

    print("[*] slor: I am a student")
    print(slor(sequence="I am a student .", lm=model, unigram_log_probs_=ug))
    
    print("[*] slor: student I am")
    print(slor(sequence="student I am .", lm=model, unigram_log_probs_=ug))
