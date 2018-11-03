from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from code.utils import get_labeled_toks
from code.treeops import prune
from code.utils import prune_deletes_q
from code.treeops import get_walk_from_root
from preproc.lstm_preproc import get_instance
from preproc.lstm_preproc import get_encoded_tokens
from preproc.lstm_preproc import PP
from preproc.lstm_preproc import PE
import nn.models


class NeuralNetworkTransitionGreedy:
    def __init__(self, archive_loc, query_focused=True):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.archive = archive
        self.query_focused = query_focused
        self.predictor = Predictor.from_archive(archive, "paper-classifier")

    def predict_proba(self, jdoc, vertex):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        label = "NA"
        sentence = get_labeled_toks(vertex, jdoc)
        assert "old" == "way"
        instance = self.predictor._dataset_reader.text_to_instance(sentence,
                                                                   label)
        pred = self.predictor.predict_instance(instance)
        return pred["class_probabilities"][1]

    def predict_vertexes(self, jdoc):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        if self.query_focused:
            return {_["index"]: self.predict_proba(jdoc, _["index"])
                    for _ in jdoc["tokens"] if not prune_deletes_q(_["index"],
                                                                   jdoc)}
        else:
            return {_["index"]: self.predict_proba(jdoc, _["index"])
                    for _ in jdoc["tokens"]}

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = 0
        while length != prev_length and length > int(jdoc["r"]):
            vertexes = list(self.predict_vertexes(jdoc).items())
            nops += len(vertexes)
            vertexes.sort(key=lambda x: x[1], reverse=True)
            vertex, prob = vertexes[0]
            prune(g=jdoc, v=vertex)
            prev_length = length
            length = self.get_char_length(jdoc)
        length = self.get_char_length(jdoc)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in jdoc["tokens"]]
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "nops": nops
                    }
        else:
            return {"y_pred": "could not find a compression",
                    "nops": nops
                    }


class NeuralNetworkTransitionBFS:

    def __init__(self, archive_loc, query_focused=True):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.query_focused = query_focused
        self.predictor = Predictor.from_archive(archive, "paper-classifier")

    def predict_proba(self, jdoc, vertex):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        label = "NA"
        sentence = " ".join([_["word"] for _ in get_labeled_toks(vertex, jdoc)])
        instance = self.predictor._dataset_reader.text_to_instance(sentence,
                                                                   label)
        pred = self.predictor.predict_instance(instance)
        return pred["class_probabilities"]

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

        for vertex in get_walk_from_root(original_s):
            instance = get_instance(original_s, vertex, state, oracle_label="unknown")
            if in_compression(vertex, state):
                proposed = PP
            else:
                proposed = PE
            # get_encoded_tokens(instance, state, original_s, node)
            instance["tokens"] = get_labeled_toks(vertex, state, proposed)
            instance = self.predictor._dataset_reader.text_to_instance(instance,
                                                                       "unknown")
            pred = self.predictor.predict_instance(instance)
            import ipdb;ipdb.set_trace()
