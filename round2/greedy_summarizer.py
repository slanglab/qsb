from __future__ import division
import collections
from code.treeops import prune
from code.treeops import dfs
import math

# e.g. Op(vertex=29, p_yes='.95')
Op = collections.namedtuple('Op', 'vertex p_yes deleted')


class BVCompressor(object):

    def __init__(self, predictor, r, Q, sentence, verbose=False):
        '''
        inputs:
            r: a character length constraint
            Q(list): a list of indexes
            sentence(dict): a jdoc sentence
            predictor(object): implements a predict_proba method
        '''

        if Q != []:
            words = [i["index"] for i in sentence["tokens"]]
            assert all(q in words for q in Q)
            assert type(r) is int
            assert type(sentence) is dict
        self.sentence = sentence
        self.Q = Q
        self.predictor = predictor
        self.r = r
        self.verbose = verbose
        self.history = []
        assert self.predictor.kind == "FigureEightPredictor"

    def execute_op(self, source, v):
        '''
        do an operation on source
        inputs:
            op: an option named tuple
        returns:
            none. but the operation will modify the passed object
        '''
        prune(g=source, v=v)
        self.sentence = source
        return None

    @staticmethod
    def readability(source, vertexes, predictor):
        '''
        Get a readability score from a list of vertexes
        '''
        history = []
        for v in vertexes:
            ops = list(BVCompressor.enumerate_single_op(jdoc=source, predictor=predictor, vertex=v))
            prune(g=source, v=v)
            executed = [_ for _ in ops if _.vertex == v][0]
            history.append(executed)
        return sum([math.log(_.p_yes) for _ in history])

    @staticmethod
    def worst_op(source, vertexes, predictor):
        '''
        Get a readability score from a list of vertexes
        '''
        history = []
        for v in vertexes:
            ops = list(BVCompressor.enumerate_single_op(jdoc=source, predictor=predictor, vertex=v))
            prune(g=source, v=v)
            executed = [_ for _ in ops if _.vertex == v][0]
            history.append(executed)
        return min([math.log(_.p_yes) for _ in history])

    def do_ops(self, vertexes):
        '''
        do a list of ops
        '''
        for v in vertexes:
            prune(g=self.sentence, v=v)

    def compression(self):
        return " ".join([_["word"] for _ in self.sentence["tokens"]])

    def tokens(self):
        return [_["index"] for _ in self.sentence["tokens"]]

    @staticmethod
    def enumerate_available_ops(jdoc, predictor):
        '''
        returns all of the available ops
        '''

        def get_dep(vertex_):
            return [str(x["dep"]) for x in jdoc["basicDependencies"]
                    if x["dependent"] == vertex_][0]

        for vertex in [i["index"] for i in jdoc["tokens"]]:
            dep = get_dep(vertex)
            if dep.lower() != "root":
                # workerID==0 b/c worker unk
                p_yes = predictor.predict_proba(
                                                jdoc=jdoc,
                                                op="prune",
                                                vertex=int(vertex),
                                                dep=dep,
                                                worker_id=0
                                                )
                deleted = dfs(g=jdoc, hop_s=int(vertex), D = [])
                yield Op(vertex=vertex, deleted=deleted,
                         p_yes=p_yes)

    @staticmethod
    def enumerate_single_op(jdoc, predictor, vertex):
        '''
        returns all of the available ops
        '''

        def get_dep(vertex_):
            return [str(x["dep"]) for x in jdoc["basicDependencies"]
                    if x["dependent"] == vertex_][0]

        dep = get_dep(vertex)
        if dep.lower() != "root":
            # workerID==0 b/c worker unk
            p_yes = predictor.predict_proba(
                                            jdoc=jdoc,
                                            op="prune",
                                            vertex=int(vertex),
                                            dep=dep,
                                            worker_id=0
                                            )
            deleted = dfs(g=jdoc, hop_s=int(vertex), D = [])
            yield Op(vertex=vertex, deleted=deleted,
                     p_yes=p_yes)
