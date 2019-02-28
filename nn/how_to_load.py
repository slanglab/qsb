# coding: utf-8
from models import *
from nn.models import *
from allennlp.models.archival import load_archive

arch = load_archive("/tmp/536216097", weights_file="/tmp/536216097/best.th")


