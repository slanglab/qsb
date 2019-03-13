import logging
import uuid

import socket


'''define the version of klm to use for the whole package'''
if socket.gethostname() == "dewey":
    BASEDIR = "/Users/ahandler/research/qsr"
    LOGDIR = BASEDIR + '/logs/tree.log'
elif socket.gethostname() == "hobbes":
    BASEDIR = "/home/ahandler/qsr"
    LOGDIR = BASEDIR + '/logs/tree.log'

tag = str(uuid.uuid4())[0:8]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(LOGDIR)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter(tag + '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d:\t  - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
