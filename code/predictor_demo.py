from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import nn.models
import json
archive_file = "tests/fixtures/633320059/model.tar.gz"
archive = load_archive(archive_file=archive_file)
predictor = Predictor.from_archive(archive, "paper-classifier") 

sentence = "Hello world"
label = "NA"

fn = "tests/fixtures/sample.jsonl"

with open(fn, "r") as inf:
    for _ in inf:
        _ = json.loads(_)
        sentence = " ".join(_["tokens"])
        label = _["label"]
        instance = predictor._dataset_reader.text_to_instance(sentence, label)

        # allennlp predict /mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/633320059/model.tar.gz tests/fixtures/sample.jsonl --include-package nn --predictor paper-classifier --output-file /mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/633320059/preds.jsonl && mv /mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/633320059/* tests/fixtures/633320059/

        print(predictor.predict_instance(instance))


with open("tests/fixtures/633320059/preds.jsonl", "r") as inf:
    for _ in inf:
        _ = json.loads(_)
        print(_)
