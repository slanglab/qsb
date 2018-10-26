'''
A predictor must support a predict_proba method,
which returns the probabilty that an op returns an acceptable sentence

The following predictors are what we have so far
    - FigureEightPredictor
'''
from __future__ import division

import random
import pickle


class FigureEightPredictor(object):
    '''a predictor supervised by what FigureEight workers say a sentence is'''
    def __init__(self, cache):

        with open("models/lr.p", 'r') as inf:
            self.model = pickle.load(inf)

        with open("models/featurizer.p", 'r') as inf:
            self.featurizer = pickle.load(inf)

        self.kind = "FigureEightPredictor"

    def predict_proba(self, jdoc, op, vertex, dep, worker_id=0):
        inputs = {"source_json": jdoc, "op": op, "vertex": vertex,
                  "dep": dep, "_worker_id": 0}
        feats = self.featurizer.bulk_featurize(inputs_list=[inputs]).todense().reshape(1, -1)
        return self.model.predict_proba(feats)[0][1]


