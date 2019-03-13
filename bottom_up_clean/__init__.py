import socket


'''define the version of klm to use for the whole package'''
if socket.gethostname() == "dewey":
    BASEDIR = "/Users/ahandler/research/qsr"
    LOGDIR = BASEDIR + '/logs/tree.log'
elif socket.gethostname() == "hobbes":
    BASEDIR = "/home/ahandler/qsr"
    LOGDIR = BASEDIR + '/logs/tree.log'
