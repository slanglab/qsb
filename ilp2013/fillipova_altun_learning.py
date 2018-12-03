'''
Actually does the learning for
Overcoming the Lack of Parallel Data in Sentence Compression
'''
from __future__ import division
from ilp2013.fillipova_altun import run_model
from code.log import logger
import numpy as np
import random
import pickle
import argparse
from sklearn.metrics import f1_score
from code.printers import pretty_print_conl
from ilp2013.fillipova_altun_supporting_code import *
from code.utils import get_NER_query
from code.utils import get_gold_y,get_pred_y

random.seed(1)


def learn(dataset, vocab, epsilon=1, epochs=20, start_epoch=1, verbose=False, snapshot=False):
    '''
    Do learning for a dataset of compression data
    '''
    with open("checkpoints/latest", "rb") as of:
        checkpoint = pickle.load(of)

    weights = checkpoint["weights"]
    avg_weights = checkpoint["avg_weights"]
    dataset_queue = checkpoint["dataset_queue"]
    t = checkpoint["t"]
    start_epoch = checkpoint["epoch"]

    print("[*] running on ", len(dataset))

    for epoch in range(start_epoch, epochs):
        if verbose:
            print("[*] epoch {}".format(epoch))
        epoch_scores = []
        random.shuffle(dataset_queue)
        while len(dataset_queue) > 0:
            t += 1
            source_jdoc = dataset[dataset_queue[0]]
            dataset_queue = dataset_queue[1:]
            random.shuffle(dataset_queue)
            gold = get_gold_edges(source_jdoc)

            # "The maximum permitted compression length is set to be the same as the
            # length of the oracle compression" => F & A

            r = get_oracle_r(source_jdoc)
            # Q = get_NER_query(source_jdoc)
            # print Q

            # So if you tell the model Q, it seems to always pick edges between
            # the root and the query tokens which are inserted via the Fillipova tree
            # transform. This leads to worse F1 (edge and token level)
            # I think it is fine to just run the F & A model as stated in their
            # paper (no query term). I was thinking to add query term thinking
            # it would be a way to be more generous to the baseline but in fact
            # the query term is making the model worse so no query. Sorting out
            # and fixing this would change the model they propose in their paper, which
            # is then no longer a baseline.

            # page 5 of their paper.
            # "The maximum permitted compression
            # length is set to be the same as the length
            # of the oracle compression" => so r param below is appropriate
            try:
                output = run_model(source_jdoc, vocab=vocab, weights=weights, r=r)
            except IndexError:
                output = {"solved": False}
                with open("/tmp/confusion","wb") as of:
                    debug = (source_jdoc, vocab, weights, r)
                    pickle.dump(debug, of)
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

                original_indexes = [_["index"] for _ in source_jdoc["tokens"]]
                y_gold = get_gold_y(source_jdoc)
                predicted_compression = [o['dependent'] for o in output["get_Xs"]]
                y_pred = get_pred_y(predicted_compression=predicted_compression,
                                    original_indexes=original_indexes)

                epoch_scores.append(f1_score(y_true=y_gold, y_pred=y_pred))
            if verbose:
                print(f1_score(y_true=y_gold, y_pred=y_pred))
            if (t % 1000 == 0):
                logger.info("{}-{}-{}".format(np.mean(epoch_scores), t, epoch))
                epoch_scores = []
                with open("checkpoints/latest", "wb") as of:
                    checkpoint = {"weights": weights,
                                  "t": t,
                                  "epoch": epoch,
                                  "avg_weights": avg_weights,
                                  "dataset_queue": dataset_queue}
                    pickle.dump(checkpoint, of)
        if snapshot:
            with open("snapshots/{}".format(epoch), "wb") as of:
                pickle.dump(avg_weights, of)

    return {"avg_weights": avg_weights, "final_weights": weights}


def init_checkpoints(dataset, vocab):
    dataset_queue = list(range(len(dataset)))
    random.shuffle(dataset_queue)
    with open("checkpoints/latest", "wb") as of:
        checkpoint = {"weights": zero_weights(vocab),
                      "t": 0,
                      "epoch": 0,
                      "avg_weights": zero_weights(vocab),
                      "dataset_queue": dataset_queue}
        pickle.dump(checkpoint, of)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', nargs="?", default=20, type=int)
    parser.add_argument("-file", default="preproc/100k")
    args = parser.parse_args()
    vocab = get_all_vocabs()
    with open(args.file, "rb") as of:
        data = pickle.load(of)

    #init_checkpoints(data, vocab)

    averaged_weights = learn(dataset=data, vocab=vocab, snapshot=True,
                             epochs=args.epochs, verbose=False)
    with open("output/{}".format(args.epochs), "wb") as of:
        pickle.dump(averaged_weights, of)
