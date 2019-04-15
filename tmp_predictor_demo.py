# note! 
# export PYTHONPATH=.:allennlp2/allennlpallennlpasalibraryexample/
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

from allennlp2.allennlpallennlpasalibraryexample.my_library.models import *
from allennlp2.allennlpallennlpasalibraryexample.my_library.dataset_readers import *
from allennlp2.allennlpallennlpasalibraryexample.my_library.predictors import *

import json
archive_file = "models/457282793/model.tar.gz"
archive = load_archive(archive_file=archive_file)
predictor = Predictor.from_archive(archive, "paper-classifier") 


fn = "allennlp2/allennlpallennlpasalibraryexample/tests/fixtures/s2_papers.jsonl"

with open(fn, "r") as inf:
    for _ in inf:
        _ = json.loads(_)
        print(_.keys())
        sentence = _["sentence"] 
        label = _["label"]
        instance = predictor._dataset_reader.text_to_instance(sentence, label)
        print(predictor.predict_instance(instance))


