import socket
import json
import kenlm as klm

'''define the version of klm to use for the whole package'''
if socket.gethostname() == "dewey":
    BASEDIR = "/Users/ahandler/qsr"
    klm = klm.LanguageModel(BASEDIR + '/klm/vox.klm')
    with open(BASEDIR + "/klm/unigram.json", "r") as inf: # use giga word for these b/c not computered w/ vox
        unigram_probs = json.load(inf)
    LOGDIR = BASEDIR + '/logs/tree.log'
    COLLOCATION_DIRECTORY = "collocation_lookups/"
elif socket.gethostname() == "hobbes":
    BASEDIR = "/home/ahandler/qsr"
    klm = klm.LanguageModel(BASEDIR + '/klm/gigaword.klm')
    ### Important! this variable is poorly named.
    ### These are unigram log probs !
    with open(BASEDIR + "/klm/unigram.json", "r") as inf:
        unigram_probs = json.load(inf)
    LOGDIR = BASEDIR + '/logs/tree.log'
    COLLOCATION_DIRECTORY = "collocation_lookups/"
    PICK = "can_sentence_1_be_shortened_to_sentence_2"



