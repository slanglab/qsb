
# coding: utf-8
import pdb
from models import *
from nn.models.bottom_up_simple import *
from nn.dataset_readers.bottom_up_reader import *
from nn.predictors.bottom_up_predictor import *
from nn.models import *
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

loc = "/tmp/548079730"
arch = load_archive(loc, weights_file=loc + "/best.th")
predictor_name = "bottom_up_predictor"

predictor = Predictor.from_archive(arch, predictor_name)

pdb.set_trace()
