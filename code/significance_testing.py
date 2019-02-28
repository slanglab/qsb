from __future__ import division
import random
import numpy as np
import json
from code.log import logger
from tqdm import tqdm


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


def filelist2datadict(list_):
    return {_: json.load(open(_)) for _ in files_to_load}


if __name__ == "__main__":

    files_to_load = ["comp_experiments_f1/output/full-556251071-nn-prune-greedy-test",
                     "comp_experiments_f1/output/full-fixtures-ilp7-test"]

    dt = filelist2datadict(files_to_load)

    pairs_we_care_about = []

    pairs_we_care_about.append((files_to_load[0], files_to_load[1]))

    for _ in pairs_we_care_about:
        file_one, file_two = _
        if file_one != file_two:
            one = [dt[file_one][i]["f1"] for i in dt[file_one] if "sentence" in i]
            two = [dt[file_two][i]["f1"] for i in dt[file_two] if "sentence" in i]
            f1_1 = np.mean(one)
            f1_2 = np.mean(two)
            bigger, smaller = (one, two) if f1_1 > f1_2 else (two, one)
            bigger_file, smaller_file = (file_one, file_two) if f1_1 > f1_2 else (file_two, file_one)
            assert bigger != smaller
            params = SamplingParameters(model_one_data=bigger,
                                        model_two_data=smaller,
                                        f=np.mean,
                                        b=10000)
            pv, samples = run_b_samples(params)
            logger.info("sig p" + ",".join([bigger_file,
                                            smaller_file,
                                            str(pv),
                                            str(params.delta_x)]))
