#coding: utf-8

from __future__ import print_function 
import cPickle as pickle
from tqdm import tqdm
from sklearn.metrics import f1_score

with open("tests/fixtures/mini_proc", "r") as of:
    training_data = pickle.load(of)[0:100]

from ilp2013.fillipova_altun import run_model
from ilp2013.fillipova_altun_supporting_code import *
from code.utils import get_NER_query,get_gold_y,get_pred_y

vocab = get_all_vocabs()
weights = zero_weights(vocab)

for jdoc in tqdm(training_data):
    Q = jdoc["compression_indexes"]
    y_gold = get_gold_y(jdoc)
    original_indexes = [_["index"] for _ in jdoc["tokens"]]
    r = get_oracle_r(jdoc)
    output = run_model(jdoc, vocab=vocab, weights=weights, r=r, Q=Q)
    assert (all([o in [_["dependent"] for _ in output["get_Xs"]] for o in Q]))
    predicted_compression = [o['dependent'] for o in output["get_Xs"]]
    y_pred = get_pred_y(predicted_compression=predicted_compression,
                        original_indexes=original_indexes)
    print(f1_score(y_true=y_gold, y_pred=y_pred))

for jdoc in tqdm(training_data):
    Q = get_NER_query(jdoc)
    r = get_oracle_r(jdoc)
    output = run_model(jdoc, vocab=vocab, weights=weights, r=r, Q=Q)
    assert (all([o in [_["dependent"] for _ in output["get_Xs"]] for o in Q]))


