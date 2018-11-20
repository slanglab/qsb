from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('split_classifier')
class BinaryPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = " ".join([i["word"] for i in json_dict['tokens']])

        if json_dict["label"] not in ["p", "e"]:
            label = "1"
        else:
            label = "0"
        if "p" in json_dict["label"]:
            is_prune = True
        else:
            is_prune = False

        instance = self._dataset_reader.text_to_instance(sentence=sentence,
                                                        label=label,
                                                        is_prune=is_prune)
        return instance
