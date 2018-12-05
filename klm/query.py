from __future__ import division
from sklearn import linear_model

import kenlm

LOC = "/home/ahandler/qsr/klm/fa.klm"


class LM:

    def __init__(self):

        self.model = kenlm.LanguageModel(LOC)

    def score(self, str_):
        # str_ is a " "-delimited string, e.g. "I am a student"

        return self.model.score(str_)


if __name__ == "__main__":

    model = LM()

    print(model.score("I am a student"))
    print(model.score("student I a am"))
