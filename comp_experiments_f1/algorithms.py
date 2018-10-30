from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from code.utils import get_labeled_toks
import nn.models
import json


class NeuralNetworkTransitionBasedBFS:
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
        return pred["class_proabilities"][1]
