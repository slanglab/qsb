from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import nn.models
archive_file = "/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/633320059/model.tar.gz"
archive = load_archive(archive_file=archive_file)
predictor = Predictor.from_archive(archive, "paper-classifier") 

sentence = "Hello world"
label = "NA"

instance = predictor._dataset_reader.text_to_instance(sentence, label)

print(predictor.predict_instance(instance))
