'''
Actually does the learning for
Overcoming the Lack of Parallel Data in Sentence Compression
'''
from __future__ import division
from ilp2013.fillipova_altun import run_model
from code.log import logger
import glob
import os
import random
import pickle
import string
import ujson as json
import argparse
from code.printers import pretty_print_conl
from ilp2013.fillipova_altun_supporting_code import *
from code.utils import get_NER_query


random.seed(1)


def learn(dataset, vocab, epsilon=1, epochs=20, verbose=False, snapshot=False):
    '''
    Do learning for a dataset of compression data
    '''
    weights = zero_weights(vocab)
    avg_weights = np.copy(weights) # a running average of the weights
    t = 0
    print "[*] running on ", len(dataset)

    for epoch in range(1, epochs):
        if verbose:
            print "[*] epoch {}".format(epoch)
        random.shuffle(dataset)
        epoch_scores = []
        for d in dataset:
            t += 1
            source_jdoc = d
            gold = get_gold_edges(source_jdoc)
            r = get_oracle_r(source_jdoc)
            Q = get_NER_query(source_jdoc)
            #The maximum permitted compression length is set to be the same as the length of the oracle compression
            print Q
            output = run_model(source_jdoc, vocab=vocab, weights=weights, r=r, Q=Q)
            if output["solved"]:
                gold.sort()
                pred = output["predicted"]
                pred.sort()
                weights = non_averaged_update(gold=gold, predicted=output["predicted"],
                                              w_t=weights, vocabs=vocab, jdoc=source_jdoc,
                                              epsilon=epsilon)
                avg_weights *= (t - 1)
                avg_weights += weights
                avg_weights /= t
                epoch_scores.append(f1(output["predicted"], gold))
            if verbose:
                print f1(output["predicted"], gold)
            if (t % 1000 == 0):
                logger.info("{}-{}-{}".format(np.mean(epoch_scores), t, epoch))
                epoch_scores = []
        if snapshot:
            with open("snapshots/{}".format(epoch), "w") as of:
                pickle.dump(avg_weights, of)
    return {"avg_weights":avg_weights, "final_weights":weights}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', nargs="?", default=None, type=int)
    parser.add_argument('-epochs', nargs="?", default=20, type=int)
    args = parser.parse_args()
    vocab = get_all_vocabs()
    with open("preproc/100k", "r") as of:
        data = pickle.load(of)
    if args.N is not None:
        data = data[0:args.N]
    averaged_weights = learn(dataset=data, vocab=vocab, snapshot=True, epochs=args.epochs)
    with open("output/{}".format(args.epochs), "w") as of:
        pickle.dump(averaged_weights, of)
