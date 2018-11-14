'''
All of the algorithms for query-focused compression
python 3
'''

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from code.treeops import prune
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
import numpy as np
import copy
import nn
import nn.models

from allennlp.common.util import import_submodules
import_submodules('nn')

class NeuralNetworkTransitionGreedy:
    def __init__(self, archive_loc, model_name, predictor_name, query_focused=True):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.archive = archive
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
            if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
                best = v
        return best

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = 0
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
        while length != prev_length and length > int(jdoc["r"]):
            vertexes = list(self.predict_vertexes(jdoc=jdoc, state=state).items())
            nops += len(vertexes)
            vertexes.sort(key=lambda x: x[1], reverse=True)
            if len(vertexes) == 0:
                print("huh")
                break
            vertex, prob = vertexes[0]
            prune(g=state, v=vertex)
            prev_length = length
            length = self.get_char_length(state) 
        length = self.get_char_length(state)
        if length <= int(jdoc["r"]): 
            remaining_toks = [_["index"] for _ in state["tokens"]]
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "nops": nops
                    }
        else:
            return {"y_pred": "could not find a compression",
                    "nops": nops
                    }


class NeuralNetworkTransitionBFS:

    '''
    This algorithm basically does a BFS walk and executes the greedy move

    It is mostly used to see what kind of F1 we get on oracle options
    '''

    def __init__(self, archive_loc):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.query_focused = False
        self.predictor = Predictor.from_archive(archive, "paper-classifier")
        self.archive = archive

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

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
                provisional_label = "p"
            else:
                proposed = PE
                provisional_label = "e"

            toks = get_encoded_tokens(provisional_label, state,
                                      original_s, vertex)

            txt = " ".join([_["word"] for _ in toks])

            instance = self.predictor._dataset_reader.text_to_instance(txt,
                                                                       "e")

            pred = self.predictor.predict_instance(instance)

            pred_labels = self.archive.model.vocab.get_index_to_token_vocabulary("labels")

            move = pred_labels[np.argmax(pred["class_probabilities"])]

            if move == "p":
                prune(g=state, v=vertex)
            if move == "e":
                proposed = get_proposed(original_s, vertex, state)
                state["tokens"] = proposed["tokens"]
                state["basicDependencies"] = proposed["basicDependencies"]

        remaining_toks = [_["index"] for _ in state["tokens"]]

        return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                "nops": len(original_s["tokens"])
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

        predicted_compression = [o['dependent'] for o in output["get_Xs"]]
        y_pred = get_pred_y(predicted_compression=predicted_compression,
                            original_indexes=original_indexes)

        return {"y_pred": y_pred,
                "nops": -19999999  # whut to do here????
                }


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
