from __future__ import division
import random
import numpy as np
import glob
import itertools
import json
from code.log import logger
from tqdm import tqdm
from round2.model2 import get_roc_auc_from_datapoints


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
    pairs = zip(params.model_one_data, params.model_two_data)
    bootstrap_world = boostrap(pairs)
    model_one = params.f([_[0] for _ in bootstrap_world])
    model_two = params.f([_[1] for _ in bootstrap_world])
    return model_one - model_two


def run_b_samples(params):
    samples = [run_sample(params) for _ in tqdm(range(params.b))]
    pval = np.mean([_ > params.delta_x * 2 for _ in samples])
    return pval, samples


if __name__ == "__main__":

    fns = glob.glob("output/*multiop*xval")

    files_to_load = list(fns) + ["output/test_full",
                                 "output/test_full-dep",
                                 "output/test_full-worker",
                                 "output/test_normlp",
                                 "output/test_cola",
                                 "output/test_normlp+dep",
                                 "output/test_normlp+worker"]

    dt = {_: json.load(open(_)) for _ in files_to_load}

    # pairs_we_care_about = list(itertools.product(fns, fns))
    # running this all pairs  takes to long and we only care about these 5

    pairs_we_care_about = []

    pairs_we_care_about.append(("output/test_full", "output/test_full-dep"))

    pairs_we_care_about.append(("output/test_full", "output/test_full-worker"))

    pairs_we_care_about.append(("output/test_full", "output/test_normlp"))

    pairs_we_care_about.append(("output/test_full", "output/test_cola"))

    pairs_we_care_about.append(("output/test_full", "output/test_normlp+dep"))

    pairs_we_care_about.append(("output/test_full", "output/test_normlp+worker"))

    for _ in pairs_we_care_about:
        f1, f2 = _
        print f1, f2
        if f1 != f2:
            one = dt[f1]
            two = dt[f2]
            auc_1 = get_roc_auc_from_datapoints(one)
            auc_2 = get_roc_auc_from_datapoints(two)
            bigger = f1 if auc_1 > auc_2 else f2
            smaller = f2 if auc_1 > auc_2 else f1
            assert bigger != smaller
            params = SamplingParameters(model_one_data=dt[bigger],
                                        model_two_data=dt[smaller],
                                        f=get_roc_auc_from_datapoints,
                                        b=10000)
            pv, samples = run_b_samples(params)
            logger.info("sig p" + ",".join([bigger,
                                            smaller,
                                            str(pv),
                                            str(params.delta_x)]))
