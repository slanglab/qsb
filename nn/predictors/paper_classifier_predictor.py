from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('paper-classifier')
class PaperClassifierPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = " ".join(json_dict['tokens'])
        instance = self._dataset_reader.text_to_instance(sentence=sentence)
        return instance
