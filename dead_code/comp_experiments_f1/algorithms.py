'''
All of the algorithms for query-focused compression
python 3
'''
import math
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from code.treeops import prune
from code.log import logger
from code.utils import prune_deletes_q
from code.treeops import get_walk_from_root
from code.utils import get_pred_y
from preproc.lstm_preproc import get_encoded_tokens
from preproc.lstm_preproc import get_proposed
from ilp2013.fillipova_altun import run_model
from code.treeops import dfs
from ilp2013.fillipova_altun_supporting_code import filippova_tree_transform
from preproc.lstm_preproc import PP
from preproc.lstm_preproc import PE
from code.printers import pretty_print_conl
from math import log
from random import shuffle
import numpy as np
import copy
import nn
import nn.models

from allennlp.common.util import import_submodules
import_submodules('nn')


class NeuralNetworkTransitionGreedy:
    def __init__(self, archive_loc, model_name, predictor_name, T=.5, query_focused=True):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.archive = archive
        self.T = T
        self.query_focused = query_focused
        self.predictor = Predictor.from_archive(archive, predictor_name)

    def predict_proba(self, original_s, vertex, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        provisional_label = "p"
        toks = get_encoded_tokens(provisional_label, state,
                                  original_s, vertex)

        txt = " ".join([_["word"] for _ in toks])

        instance = self.predictor._dataset_reader.text_to_instance(txt, True,
                                                                   "1")

        pred_labels = self.archive.model.vocab.get_index_to_token_vocabulary("labels")
        op2n = {v:k for k,v in pred_labels.items()}
        pred = self.predictor.predict_instance(instance)
        return pred["class_probabilities"][op2n["1"]]

    def predict_vertexes(self, jdoc, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        assert self.query_focused
        return {_["index"]: self.predict_proba(original_s=jdoc,
                                               vertex=_["index"],
                                               state=state)
                for _ in state["tokens"] if not prune_deletes_q(_["index"],
                                                                jdoc)}

    def get_char_length(self, jdoc):
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def heuristic_extract(self, jdoc):
        '''
        return the lowest vertex in the tree that contains the query terms
        '''
        from_root = [_['dependent'] for _ in jdoc["basicDependencies"] if _['governor'] == 0][0]
        best = from_root
        def tok_is_verb(vertex):
            gov = [o["pos"][0] for o in jdoc["tokens"] if o["index"] == v][0]
            return gov[0].lower() == "v"
        for v in get_walk_from_root(jdoc):  # bfs
            children = dfs(g=jdoc, hop_s=v, D=[])
            # the verb heuristic is b/c the min governing tree is often just Q itself
            if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
                best = v
        return best

    def init_state(self, jdoc):
        '''init to the governing subtree'''
        topv = self.heuristic_extract(jdoc)
        if jdoc["q"] != []:
            short_tree = dfs(g=jdoc, hop_s=topv, D=[])
            toks_to_start = [i for i in jdoc["tokens"] if i["index"] in short_tree]
            deps_to_start = [i for i in jdoc["basicDependencies"] if
                             i["dependent"] in short_tree
                             and i["governor"] in short_tree]
            state = {"tokens": toks_to_start, "basicDependencies": deps_to_start}
        else:
            state = {"tokens": jdoc["tokens"], "basicDependencies": jdoc["basicDependencies"]}

        # this line is needed for the "For roughly two-thirds of sentences ..."
        # logger.info("V init sentence ", len(jdoc["tokens"]) == len(state["tokens"]))

        # this line is needed for the "In more than 95% of sentences, the compression system..."
        #r = " ".join([o["word"] for o in state["tokens"]])
        #logger.info("init > r {}".format(len(r) > int(jdoc["r"])))
        return state

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = []

        state = self.init_state(jdoc)
        prunes = 0
        while length != prev_length and length > int(jdoc["r"]):
            prunes += 1
            vertexes = list(self.predict_vertexes(jdoc=jdoc, state=state).items())
            nops.append(len(vertexes))
            vertexes.sort(key=lambda x: x[1], reverse=True)
            if len(vertexes) == 0:
                print("ending pruning")
                pretty_print_conl(jdoc) 
                break # this is a case where there is no compression. there are no vertexes
                      # that you can prune w/o removing a query term
            vertex, prob = vertexes[0]
            prune(g=state, v=vertex)
            prev_length = length
            length = self.get_char_length(state)
        length = self.get_char_length(state)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in state["tokens"]]
            compression = " ".join([_["word"] for _ in jdoc["tokens"] if _["index"] in remaining_toks])
            assert all([o in remaining_toks for o in jdoc["q"]]) 
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "compression": compression,
                    "nops": nops,
                    "prunes": prunes
                    }
        else:
            # in rare cases (5 of 1000) this method will not be able to prune 
            # to below the length constraint. In these cases just return q.
            # TODO add to paper
            return {"y_pred": [_ in jdoc["q"] for _ in orig_toks],  
                    "nops": nops
                    }


class FMCSearch:
    def __init__(self, archive_loc, model_name,
                 predictor_name,nsamples):
        assert type(archive_loc) == str
        self.nsamples = nsamples
        self.predictor = NeuralNetworkTransitionGreedy(archive_loc,
                                                       model_name,
                                                       predictor_name,
                                                       query_focused=True)

    def run_one(self, jdoc_sentence):
        jdoc_sentence = copy.deepcopy(jdoc_sentence)
        state = self.predictor.init_state(jdoc_sentence)
        length = self.predictor.get_char_length(state)
        orig_toks = jdoc_sentence["original_ix"]
        score = 0
        nops = 0
        prunes = 0
        while length > jdoc_sentence["r"]:
            v2prob = self.predictor.predict_vertexes(jdoc_sentence, state)
            nops += len(v2prob)
            prunes += 1
            total_probability = sum(v2prob.values())

            # list on dict keys/vals returns in insertion order, i.e. consistent order
            distribution_ops = np.asarray([(p/total_probability) for
                                          p in list(v2prob.values())])
            # probability over ## indexes
            pick = np.random.choice(len(v2prob), size=None, p=distribution_ops)
            vertex = list(v2prob.keys())[pick]
            prune(g=state, v=vertex)
            score += log(v2prob[vertex], 10)
            length = self.predictor.get_char_length(state)

        remaining_toks = [_["index"] for _ in state["tokens"]]

        return {"score": score,
                "y_pred": [_ in remaining_toks for _ in orig_toks],
                "nops": nops,
                "prunes": prunes
                }

    def predict(self, jdoc):
        options = [self.run_one(jdoc) for i in range(self.nsamples)]
        options.sort(key=lambda x: x["score"], reverse=True)
        return options[0]


class BaselineCompressor:

    '''
    This implements a baseline that just guesses the query terms
    '''

    def __init__(self):
        pass

    def predict(self, original_s):
        '''
        baseline prediction
        '''

        original_indexes = [_["index"] for _ in original_s["tokens"]]
        y_pred = get_pred_y(predicted_compression=original_s["q"],
                            original_indexes=original_indexes)

        return {"y_pred": y_pred,
                "nops": 0  # whut to do here????
                }


class FA2013Compressor:

    '''
    This implements a query query_focused compression w/ F and A
    '''

    def __init__(self, weights):
        from ilp2013.fillipova_altun_supporting_code import get_all_vocabs
        self.weights = weights
        self.vocab = get_all_vocabs()

    def predict(self, original_s):
        '''
        run the ILP
        '''

        r = int(original_s["r"])

        original_indexes = [_["index"] for _ in original_s["tokens"]]

        transformed_s = filippova_tree_transform(copy.deepcopy(original_s))

        output = run_model(transformed_s,
                           vocab=self.vocab,
                           weights=self.weights,
                           Q=original_s["q"],
                           r=r)

        # note: not necessarily a unique set b/c multi edges possible to same
        # vertex after the transform. Set() should not affect eval, if you look
        # at the code in get_pred_y
        predicted_compression = set([o['dependent'] for o in output["get_Xs"]])
        y_pred = get_pred_y(predicted_compression=predicted_compression,
                            original_indexes=original_indexes)

        assert all([i in predicted_compression for i in original_s["q"]])
        assert len(output["compressed"]) <= original_s["r"]
        return {"y_pred": y_pred,
                "compression": output["compressed"],
                "nops": -19999999  # whut to do here????
                }


class FA2013CompressorStandard:

    '''
    This implements a standard version of F and A
    '''

    def __init__(self, weights):
        from ilp2013.fillipova_altun_supporting_code import get_all_vocabs
        self.weights = weights
        self.vocab = get_all_vocabs()

    def predict(self, original_s):
        '''
        run the ILP
        '''

        original_indexes = [_["index"] for _ in original_s["tokens"]]

        transformed_s = filippova_tree_transform(copy.deepcopy(original_s))

        # "To get comparable
        # results, the unsupervised and our systems used
        # the same compression rate: for both, the requested
        # maximum length was set to the length of the headline."
        r = len(original_s["headline"])

        output = run_model(transformed_s,
                           vocab=self.vocab,
                           weights=self.weights,
                           r=r)

        predicted_compression = [o['dependent'] for o in output["get_Xs"]]
        y_pred = get_pred_y(predicted_compression=predicted_compression,
                            original_indexes=original_indexes)

        return {"y_pred": y_pred,
                "compression": output["compressed"],
                "nops": -19999999  # whut to do here????
                }

# This one is mostly a technical curiousity.
# Just a transition-based compressor that does
# traditional sentence compression
class NeuralNetworkTransitionBFS:

    '''
    This algorithm basically does a BFS walk and executes the greedy move

    It is mostly used to see what kind of F1 we get on oracle options
    '''

    def __init__(self, archive_loc, model_name, predictor_name, T=.5):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.archive = archive
        self.T = T
        self.predictor = Predictor.from_archive(archive, predictor_name)

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def predict_proba(self, original_s, vertex, state, is_prune):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        if is_prune:
            provisional_label = "p"
        else:
            provisional_label = "e"
        toks = get_encoded_tokens(provisional_label, state,
                                  original_s, vertex)

        txt = " ".join([_["word"] for _ in toks])

        instance = self.predictor._dataset_reader.text_to_instance(txt,
                                                                   is_prune,
                                                                   "1")

        pred_labels = self.archive.model.vocab.get_index_to_token_vocabulary("labels")
        op2n = {v: k for k, v in pred_labels.items()}
        pred = self.predictor.predict_instance(instance)
        return pred["class_probabilities"][op2n["1"]]

    def predict(self, original_s):
        '''
        return a compression that preserves q and respects r
        '''

        state = {"tokens": [], "basicDependencies": []}

        def in_compression(vertex, state_):
            return vertex in [i["index"] for i in state_["tokens"]]

        orig_toks = original_s["original_ix"]

        for vertex in get_walk_from_root(original_s):
            if in_compression(vertex, state):
                proposed = PP
            else:
                proposed = PE

            is_prune = proposed == PP

            prob = self.predict_proba(original_s, vertex, state, is_prune)

            if is_prune and prob > self.T:
                prune(g=state, v=vertex)
            if not is_prune and prob > self.T:
                proposed = get_proposed(original_s, vertex, state)
                state["tokens"] = proposed["tokens"]
                state["basicDependencies"] = proposed["basicDependencies"]

        remaining_toks = [_["index"] for _ in state["tokens"]]

        return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                "nops": len(original_s["tokens"])
                }


class WorstCaseCompressor:
    '''
    Performs the absolutely worst number of ops

    Prunes only a singleton vertex
    '''
    def __init__(self):
        pass

    def predict_len(self, original_s, vertex, state):
        '''
        the length is just the number of children
        '''
        children = [o for o in state["basicDependencies"]
                    if o["governor"] == vertex]
        return len(children)

    def predict_vertexes(self, jdoc, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        return {_["index"]: self.predict_len(original_s=jdoc,
                                             vertex=_["index"],
                                             state=state)
                for _ in state["tokens"] if not prune_deletes_q(_["index"],
                                                                jdoc)}

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def heuristic_extract(self, jdoc):
        '''
        return the lowest vertex in the tree that contains the query terms
        '''
        from_root = [_['dependent'] for _ in jdoc["basicDependencies"] if _['governor'] == 0][0]
        best = from_root
        def tok_is_verb(vertex):
            gov = [o["pos"][0] for o in jdoc["tokens"] if o["index"] == v][0]
            return gov[0].lower() == "v"
        for v in get_walk_from_root(jdoc):  # bfs
            children = dfs(g=jdoc, hop_s=v, D=[])
            # the verb heuristic is b/c the min governing tree is often just Q itself
            if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
                best = v
        return best

    def init_state(self, jdoc):
        '''init to the governing subtree'''
        topv = self.heuristic_extract(jdoc)
        if jdoc["q"] != []:
            short_tree = dfs(g=jdoc, hop_s=topv, D=[])
            toks_to_start = [i for i in jdoc["tokens"] if i["index"] in short_tree]
            deps_to_start = [i for i in jdoc["basicDependencies"] if
                             i["dependent"] in short_tree
                             and i["governor"] in short_tree]
            state = {"tokens": toks_to_start, "basicDependencies": deps_to_start}
        else:
            state = {"tokens": jdoc["tokens"], "basicDependencies": jdoc["basicDependencies"]}

        # this line is needed for the "For roughly two-thirds of sentences ..."
        # logger.info("V init sentence ", len(jdoc["tokens"]) == len(state["tokens"]))

        # this line is needed for the "In more than 95% of sentences, the compression system..."
        #r = " ".join([o["word"] for o in state["tokens"]])
        #logger.info("init > r {}".format(len(r) > int(jdoc["r"])))
        return state

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = []

        state = self.init_state(jdoc)
        prunes = 0
        while length != prev_length and length > int(jdoc["r"]):
            prunes += 1
            vertexes = list(self.predict_vertexes(jdoc=jdoc, state=state).items())
            nops.append(len(vertexes))
            vertexes = [o for o in vertexes if o[1] == 0]
            if len(vertexes) > 0:
                shuffle(vertexes)
                vertex, length = vertexes[0]
                assert length == 0
                prune(g=state, v=vertex)
                prev_length = length
                length = self.get_char_length(state)
            else:
                break
        length = self.get_char_length(state)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in state["tokens"]]
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "nops": nops,
                    "prunes": prunes
                    }
        else:
            return {"y_pred": "could not find a compression",
                    "nops": nops
                    }
