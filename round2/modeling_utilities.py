'''
Utilities for modeling
'''
from __future__ import division

from sklearn.pipeline import FeatureUnion

import numpy as np
import pickle
import sys
import json
import itertools
import round2.features
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.sparse import hstack
from round2.dataset import PICK
from _jsonnet import evaluate_file
from round2.dataset import *
from inspect import getmembers, isfunction

sys.path.insert(0,'..')


class FeaturizerConfigs(object):
    '''
    Note: by default the featuizer will include per_dep_cv in the cv_features
          this is because you want this to play the role of the "intercept" in a 
          dependency-only model
    '''
    def __init__(self,
                 standalone_features, cv_features, include_worker_features=None):
        self.standalone_features = standalone_features
        self.cv_features = cv_features


def load_cached_per_dependency_training_set(dependency, dir_="cache/"):
    with open(dir_ + "ds.{}.p".format(dependency), "r") as inf:
        return pickle.load(inf)


def read_cache_json(fn):
    with open(fn, "r") as inf:
        return json.load(inf)


def read_from_cache(dir_="cache/"):

    def read_yes_rates_from_cache(dir_="cache/"):
        out = {}
        with open(dir_ + "dep_yes_rates.txt", "r") as inf:
            for _ in inf:
                dep, prob, count = _.split()
                out[dep] = float(prob)
        return out

    out = {
            "unigram": read_cache_json(dir_ + "vocab.json"),
            "pos": read_cache_json(dir_ + "pos.json"),
            "dep": read_cache_json(dir_ + "dep.json"),
            "workers": read_cache_json(dir_ + "workers.json"),
            "endings_3": read_cache_json(dir_ + "endings_3.json"), 
            "worker_yes_rate": read_cache_json(dir_ + "yes_rate.json"),
            "per_dep_yes_rates": read_yes_rates_from_cache(dir_=dir_)
            }
    return out


def get_featurizer(features, dir_="cache/"):
    cache = read_from_cache(dir_=dir_)
    configs = FeaturizerConfigs(cv_features=features['cv_features'],
                                standalone_features=features['standalone_features'])

    return BulkFeatureizer(cache, configs)


def get_X_and_y(dataset, featurizer):
    X = featurizer.bulk_featurize(dataset)
    y = np.asarray([_[PICK] for _ in dataset])
    return X, y


def get_feature_handle(feature_name):
    '''
    get a handle to a feature in round2.features

    feature_name should be the name of a function in round2.features
    '''
    return [o[1] for o in getmembers(round2.features) if isfunction(o[1]) and o[0] == feature_name][0]


def read_features(dep, feature_kind, config_file="round2/configs.jsonnet"):
    assert feature_kind in ["standalone_features", "cv_features"]
    return [get_feature_handle(_) for _ in json.loads(evaluate_file(config_file))[dep][feature_kind]]

def read_deps(config_file="round2/configs.jsonnet"):
    return json.loads(evaluate_file(config_file)).keys()


def read_global_cvs(config_file="round2/globals.jsonnet"):
    return [get_feature_handle(_) for _ in json.loads(evaluate_file(config_file))["cvs"]]


def read_global_standalones(config_file="round2/globals.jsonnet"):
    return [get_feature_handle(_) for _ in json.loads(evaluate_file(config_file))["binaries"]]


def get_coef(featurizer, model):
    '''get a dictionary of named coefficients'''
    assert len(featurizer.names()) == len(model.coef_[0])
    return {k: v for k, v in zip(featurizer.names(), model.coef_[0])}


class CrossValidator(object):

    def _gen_folds(self):

        unit_ids = set([_["_unit_id"] for _ in self.data])
        unit_id_to_integer = {unitid: n for n, unitid in enumerate(unit_ids)}

        for foldno in range(0, self.nfolds):

            test_index = [jno for jno, j in enumerate(self.data) if unit_id_to_integer[j["_unit_id"]] % self.nfolds == foldno]
            test_index.sort()

            train_index = [jno for jno, j in enumerate(self.data) if jno not in test_index]

            assert len(set(test_index) & set(train_index)) == 0
            print len(test_index), len(train_index)

            X_fold_train, X_fold_test = self.X[train_index], self.X[test_index]
            y_fold_train, y_fold_test = self.y[train_index], self.y[test_index]
            test_data = [_ for dno, _ in enumerate(self.data) if dno in test_index]
            train_data = [_ for dno, _ in enumerate(self.data) if dno in train_index]
            yield {"X_fold_train": X_fold_train,
                   "X_fold_test": X_fold_test,
                   "y_fold_train": y_fold_train,
                   "y_fold_test": y_fold_test,
                   "X_fold_test_data": test_data,
                   "X_fold_train_data": train_data}

    def run_xval(self, model):

        folds_results = defaultdict(dict)

        for foldno, fold in enumerate(self.folds):

            model.fit(fold["X_fold_train"], fold["y_fold_train"])

            train_workers = set([_["_worker_id"] for _ in fold["X_fold_train_data"]])
            test_workers = set([_["_worker_id"] for _ in fold["X_fold_test_data"]])

            worker_seen_in_train = [_["_worker_id"] in train_workers for _ in fold["X_fold_test_data"]]

            with open("/tmp/{}".format(foldno), "w") as of:
                json.dump({'train_workers': list(train_workers),
                           'test_workers': list(test_workers)}, of)

            unit_ids = [_["_unit_id"] for _ in fold["X_fold_test_data"]]
            worker_ids = [_["_worker_id"] for _ in fold["X_fold_test_data"]]
            folds_results[foldno] = {'predicted': model.predict_proba(fold["X_fold_test"]),
                                     'training_set_workers': train_workers,
                                     'test_data': fold["X_fold_test_data"],
                                     'seen': worker_seen_in_train,
                                     'unit_ids': unit_ids,
                                     'worker_ids': worker_ids,
                                     'gold': fold["y_fold_test"]}

        return folds_results

    def __init__(self, X, y, data, nfolds=5):
        self.X = X.todense()
        self.y = y
        self.data = data
        self.nfolds = nfolds
        self.folds = list(self._gen_folds())


class BulkFeatureizer(object):

    def prepend(self, cv_feat, str_):
        return cv_feat + ":" + str_

    def generate_count_vectorizer(self, list_of_things, prefix, min_df=1):
        # bugs !
        # https://stackoverflow.com/questions/42634581/countvectorizer-returns-only-zeros
        # https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

        list_of_things = [self.prepend(prefix, _) for _ in list_of_things]
        v = CountVectorizer(vocabulary=list_of_things,
                            binary=True,
                            lowercase=False, # min_df=min_df, # DOCS: "This parameter is ignored if vocabulary is not None."
                            token_pattern="f_[a-z|A-Z|_|:|0-9]+")
        return v

    def __init__(self, cache, featurizer_configs):
        '''
        featurize an entire dataset at once. The only featureizer in dataset.py
        went row by row which turned into a bottleneck as data got bigger
        '''

        self.standalone_features = featurizer_configs.standalone_features

        self.cv_features = featurizer_configs.cv_features

        from round2.features import *

        self.vectorizers = {
                            f_ending_3_amod_cv: self.generate_count_vectorizer(cache['endings_3'], 'f_ending_3_amod_cv'),
                            f_unigram_prune_aux_cv: self.generate_count_vectorizer(cache['unigram'], 'f_unigram_prune_aux_cv'),
                            f_unigram_prune_advmod_cv: self.generate_count_vectorizer(cache['unigram'], 'f_unigram_prune_advmod_cv'),
                            f_first_word_in_cut_cv: self.generate_count_vectorizer(cache['unigram'], 'f_first_word_in_cut_cv'),
                            f_unigram_prune_cv: self.generate_count_vectorizer(cache['unigram'], 'f_unigram_prune_cv'),
                            f_unigram_prune_cv_generic: self.generate_count_vectorizer(cache['unigram'], 'f_unigram_prune_cv_generic'),
                            f_unigram_prune_cv_case: self.generate_count_vectorizer(cache["unigram"], "f_unigram_prune_cv_case"),
                            f_unigram_prune_cv_cop: self.generate_count_vectorizer(cache["unigram"], "f_unigram_prune_cv_cop"),
                            f_unigram_prune_cv_neg: self.generate_count_vectorizer(cache["unigram"], "f_unigram_prune_cv_neg"),
                            f_unigram_prune_cv_acl: self.generate_count_vectorizer(cache["unigram"], "f_unigram_prune_cv_acl"),
                            f_unigram_prune_cv_dobj: self.generate_count_vectorizer(cache["unigram"], "f_unigram_prune_cv_dobj"),
                            f_unigram_prune_cv_nsubj:self.generate_count_vectorizer(cache['unigram'], 'f_unigram_prune_cv_nsubj'),
                            f_unigram_prune_cv_nmod_poss:  self.generate_count_vectorizer(cache["unigram"], "f_unigram_prune_cv_nmod_poss"),
                            f_first_pos_in_cut_cv_acl: self.generate_count_vectorizer(cache["pos"], "f_first_pos_in_cut_cv_acl"),
                            f_first_pos_in_cut_cv_dobj: self.generate_count_vectorizer(cache["pos"], "f_first_pos_in_cut_cv_dobj"),
                            f_pos_in_cut_cv_dobj: self.generate_count_vectorizer(cache["pos"], "f_pos_in_cut_cv_dobj"),
                            f_pos_in_cut_cv: self.generate_count_vectorizer(cache['pos'], 'f_pos_in_cut_cv'),
                            f_pos_out_cut_cv: self.generate_count_vectorizer(cache['pos'], 'f_pos_out_cut_cv'),
                            f_previous_pos_cv_case: self.generate_count_vectorizer(cache["pos"], "f_previous_pos_cv_case"),
                            f_last_pos_in_compression_cv: self.generate_count_vectorizer(cache['pos'], 'f_last_pos_in_compression_cv'),
                            f_previous_pos_cv: self.generate_count_vectorizer(cache['pos'], 'f_previous_pos_cv'),
                            f_pos_tag_of_determiner_parent_cv: self.generate_count_vectorizer(cache['pos'], 'f_pos_tag_of_determiner_parent_cv'),
                            f_pos_in_cut_cv_nsubj: self.generate_count_vectorizer(cache["pos"], "f_pos_in_cut_cv_nsubj"),
                            f_first_pos_in_cut_cv_nmod_poss: self.generate_count_vectorizer(cache["pos"], "f_first_pos_in_cut_cv_nmod_poss"),
                            f_pos_before_cut_cv_dobj: self.generate_count_vectorizer(cache["pos"], "f_pos_before_cut_cv_dobj"),
                            f_last_token_before_cut_cv_dobj: self.generate_count_vectorizer(cache['unigram'], "get_last_token_before_cut_cv_dobj"),
                            f_unigram_prune_cv_mark: self.generate_count_vectorizer(cache['unigram'], "f_unigram_prune_cv_mark"),
                            f_tok_after_cut_cv: self.generate_count_vectorizer(cache['pos'], "f_tok_after_cut_cv"),
                            f_g_pos_in_cut_cv: self.generate_count_vectorizer(cache['pos'], "f_g_pos_in_cut_cv"),
                            f_g_first_pos_in_cut_cv: self.generate_count_vectorizer(cache['pos'], "f_g_first_pos_in_cut_cv"),
                            f_g_ending_3_cv: self.generate_count_vectorizer(cache['endings_3'], "f_g_ending_3_cv"),
                            f_g_pos_tag_of_parent_cv: self.generate_count_vectorizer(cache['pos'], "f_g_pos_tag_of_parent_cv"),
                            f_g_previous_pos_cv: self.generate_count_vectorizer(cache['pos'], "f_g_previous_pos_cv"),
                            f_g_unigram_prune_cv: self.generate_count_vectorizer(cache['unigram'], "f_g_unigram_prune_cv"),
                            f_g_dep_cv: self.generate_count_vectorizer(cache['dep'], 'f_g_dep_cv'),
                            f_g_worker_id_cv: self.generate_count_vectorizer(cache['workers'], 'f_g_worker_id_cv'),
                            f_g_m_dep_cv: self.generate_count_vectorizer(cache['dep'], 'f_g_m_dep_cv'),
                            }

    def names(self):
        if len(self.cv_features) > 0:
            names = self.generate_cv_feature_union(self.cv_features).get_feature_names()
        else:
            names = []
        names = names + [_.__name__ for _ in self.standalone_features]
        return names

    def generate_cv_feature_union(self, cv_features):
        '''make a feature union of the cv features'''
        input_needed_for_feature_union = [(_.__name__, self.vectorizers[_]) for _ in cv_features]
        # input_needed_for_feature_union, e.g. = [("f_unigram_prune_aux_cv", cv_pos), ("f_dep_cv", cv_ug)]
        return FeatureUnion(input_needed_for_feature_union)

    def to_cv_feature_string(self, single_input):
        '''

        note: this is the bottleneck it seems. you could preprocess this but the downside I think is that then you can't try different features

        cv expects inputs like ["f_previous_pos_cv:JJ, f_unigram_prune_aux:Leo"]

        single_input is a dictionary

        so you need to run the featurizer to make the strings for the count vectorizers
        '''
        return " ".join(list(itertools.chain(*[foo(single_input) for foo in self.cv_features])))

    def standalone_featurize_one_row(self, i):
        '''
        Make a list of standalone features
        '''
        return [foo(i) for foo in self.standalone_features]

    def bulk_featurize(self, inputs_list):
        '''
        Do featurization for an entire list of inputs
        '''
        if len(self.standalone_features)> 0:
            standalones = sparse.csr_matrix(np.asarray([self.standalone_featurize_one_row(i) for i in inputs_list]))
        if len(self.cv_features) > 0:
            union = self.generate_cv_feature_union(self.cv_features)
            inputs_list_to_str = [self.to_cv_feature_string(_) for _ in inputs_list]
        if len(self.standalone_features) > 0 and len(self.cv_features) > 0:
            return hstack([union.transform(inputs_list_to_str), standalones])
        if len(self.standalone_features) == 0 and len(self.cv_features) > 0:
            return union.transform(inputs_list_to_str)
        if len(self.standalone_features) > 0 and len(self.cv_features) == 0:
            return standalones
        assert "bad" == "input"
