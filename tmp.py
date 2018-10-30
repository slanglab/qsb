from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import nn.models
archive_file = "/mnt/nfs/scratch1/ahandler/experiments/qsr/compressor/633320059/model.tar.gz"
archive = load_archive(archive_file=archive_file)
predictor = Predictor.from_archive(archive, "paper-classifier") 

