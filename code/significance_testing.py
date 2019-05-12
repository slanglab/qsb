from __future__ import division
from tqdm import tqdm
from code.log import logger
import random
import csv
import numpy as np
import json


class SamplingParameters(object):
    '''Configuration object for bootstrap sampling'''
    def __init__(self, b, f, model_one_data, model_two_data):
        '''
        b (int) = number of bootstrap samples
        f (function) = a function to compute a metric based on a dataset
        model_one_data (list): data from model 1. This has any info needed for f
        model_two_data (list): data from model 2. This has any info needed for f
        '''
        self.model_one_data = model_one_data
        self.model_two_data = model_two_data
        self.f = f
        assert len(self.model_one_data) == len(self.model_two_data)
        self.b = b
        mean_a = f(model_one_data)
        mean_b = f(model_two_data)
        self.delta_x = mean_a - mean_b
        assert self.delta_x >= 0


def boostrap(pairs):
    N = len(pairs)
    out = []
    while len(out) < N:
        out.append(random.choice(pairs))
    return out


def run_sample(params):
    '''compute delta for a bootstraped sample'''
    pairs = list(zip(params.model_one_data, params.model_two_data))
    bootstrap_world = boostrap(pairs)
    model_one = params.f([_[0] for _ in bootstrap_world])
    model_two = params.f([_[1] for _ in bootstrap_world])
    return model_one - model_two


def run_b_samples(params):
    samples = [run_sample(params) for _ in tqdm(range(params.b))]
    pval = np.mean([int(_ > (params.delta_x * 2)) for _ in samples])
    return pval, samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-file1', type=str, default="bottom_up_clean/ilp_results.jsonl")
    parser.add_argument('-file2', type=str, default='bottom_up_clean/make_decision_lr_results.jsonl')
    parser.add_argument('-metric', type=str, default='f1')
    parser.add_argument('-b', type=int, default=10000)

    args = parser.parse_args()

    one = [json.loads(i)[args.metric] for i in open(args.file1)]
    two = [json.loads(i)[args.metric] for i in open(args.file2)]

    f1_1 = np.mean(one)
    f1_2 = np.mean(two)

    if f1_1 > f1_2:
        bigger_dataset, smaller_dataset = (one, two)
        bigger_file, smaller_file = (args.file1, args.file2)
    else:
        bigger_dataset, smaller_dataset = (two, one)
        bigger_file, smaller_file = (args.file2, args.file1)

    params = SamplingParameters(model_one_data=bigger_dataset,
                                model_two_data=smaller_dataset,
                                f=np.mean,
                                b=args.b)

    pv, samples = run_b_samples(params)

    with open("bottom_up_clean/stat_sig.csv", "a") as of:
        writer = csv. writer(of)
        writer.writerow(["bigger", "smaller", "metric", "p", "delta"])
        writer.writerow([bigger_file,
                         smaller_file,
                         args.metric,
                         str(pv),
                         str(params.delta_x)])
