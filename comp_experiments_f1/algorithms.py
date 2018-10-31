from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from code.utils import get_labeled_toks
from code.treeops import prune
from code.utils import prune_deletes_q
import nn.models
import json


class NeuralNetworkTransitionGreedy:
    def __init__(self, archive_loc):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.predictor = Predictor.from_archive(archive, "paper-classifier")

    def predict_proba(self, jdoc, vertex):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        label = "NA"
        sentence = " ".join(get_labeled_toks(vertex, jdoc))
        instance = self.predictor._dataset_reader.text_to_instance(sentence,
                                                                   label)
        pred = self.predictor.predict_instance(instance)
        return pred["class_probabilities"][1]

    def predict_vertexes(self, jdoc):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        return {_["index"]: self.predict_proba(jdoc, _["index"])
                for _ in jdoc["tokens"] if not prune_deletes_q(_["index"], _)}

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]) == str
        return len(" ".join([_ for _ in jdoc["tokens"]]))

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        while length != prev_length and length > int(jdoc["r"]):
            vertexes = list(self.predict_vertexes(jdoc).items())
            vertexes.sort(key=lambda x: x[1], reverse=True)
            vertex, prob = vertexes[0]
            prune(g=jdoc, v=vertex)
            prev_length = length
            length = self.get_char_length(jdoc)
        length = self.get_char_length(jdoc)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in jdoc["tokens"]]
            return [_ in remaining_toks for _ in orig_toks]
        else:
            return "could not find a compression"
