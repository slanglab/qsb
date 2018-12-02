#!/usr/bin/python
'''
Dependency Tree Based Sentence Compression, Fillipova and Strube

and

Overcoming the lack of data in sentence compression, Fillipova and Altun

We use Gurobi for the ILP

'''
from __future__ import division
import json
import re
import argparse
from gurobipy import quicksum, GRB, Model
from ilp2013.fillipova_altun_supporting_code import get_featurized_dependency_scores
from ilp2013.fillipova_altun_supporting_code import get_q_word_and_governor


def run_model(jdoc, weights, vocab, k=1, r=100000, Q=[], verbose=False, force_debug=[]):
    '''
    Run the Fillipova and Strube model with Gurobi

    The model runs Gurobi in top k mode but you can just set k=1. This is slower
    but easier to maintain/understand

    inputs:
        jdoc(dict): a single sentence jdoc dictionary from CoreNLP
        weights(np.ndarray): the weights
        vocab(dict): a dictionary of vocabs such as dep vocab or lemma vocab
        k(int): the top k solutions to find
        r(int): max character constraint
        Q(list): a list of token indexes which must be included.

    returns:
        solution(dict): if k == 1, return a dictionary w/ correct solution info
        solutions(list of dict): if k > 1 then return solution info for top k
    '''
    assert type(Q) == list and all(type(_) == int for _ in Q)
    assert type(k) == int and k > 0
    words = jdoc["tokens"]

    edge_scores = get_featurized_dependency_scores(jdoc, vs=vocab, weights=weights)

    # Model
    m = Model("fillipova_strube")

    if not verbose:
        m.setParam('OutputFlag', 0)

    # these params are set to search for top k solutions, even when k=1
    # this is inefficient but also easier for maintence so whatever.
    # Gurobi runs slower when PoolSearchMode=2 b/c it does a grid search for
    # the best solutions but 'tevs
    # http://www.gurobi.com/documentation/8.0/examples/poolsearch_py.html#subsubsection:poolsearch.py
    m.setParam(GRB.Param.PoolSearchMode, 2)
    m.setParam(GRB.Param.PoolSolutions, k)

    '''init the X variables'''
    Xs = {}
    for relation in jdoc["enhancedDependencies"]:
        dep, gov = relation["dependent"], relation["governor"]
        Xs["{}-{}".format(gov, dep)] = m.addVar(name="{}-{}".format(gov, dep),
                                                vtype=GRB.BINARY)

    m.setObjective(sum(Xs[x] * edge_scores[x] for x in Xs), GRB.MAXIMIZE)

    # AH: query constraint
    for word_ in Q:
        if word_ != 0:
            m.addConstr(Xs[get_q_word_and_governor(word_, jdoc)] == 1)

    indexes = [_["index"] for _ in jdoc['tokens']] + [0]
    '''
    ensures each token has one head at most. this is not applicable to UD
    # equation 4 from Filippova and Strube (2008)
    for w in indexes:
        total = 0
        for h in indexes:
            if "{}-{}".format(h, w) in Xs:
                total += Xs["{}-{}".format(h, w)]
        m.addConstr(total <= 1)
    '''

    # equation 5 from the 2008 paper
    indexes_not_zero = [_["index"] for _ in jdoc["tokens"]]
    for w in indexes_not_zero:
        total = 0
        for h in indexes:
            if "{}-{}".format(h, w) in Xs:
                total += Xs["{}-{}".format(h, w)]

        total2 = 0
        for u in indexes:
            if "{}-{}".format(w, u) in Xs:
                total2 += Xs["{}-{}".format(w, u)]

        m.addConstr((total - total2/len(indexes_not_zero)) >= 0)

    '''abe's addition to model'''
    Ws = {} # only for character counting
    for w in words:
        Ws["w" + str(w["index"])] = m.addVar(name="{}".format("w" + str(w["index"])), vtype=GRB.BINARY)

    word2len = {i["index"]:len(i["word"]) for i in words}

    '''
    This is for the char contraint. If any number of parents are turned on,
    then you must turn on the Ws variable which tracks if the word is turned on
    Again, for counting characters
    '''
    for w in words:
        regex = re.compile("-{}$".format(w["index"]))
        potential_parents = [_ for _ in Xs if regex.search(_)]
        A = quicksum([Xs[_] * 1 for _ in potential_parents])
        B = quicksum([Ws["w" + str(w["index"])] * Xs[_] * 1 for _ in potential_parents])
        m.addConstr(A == B)


    # equation 6 from the 2008 paper. the \alpha var in the paper is equivlent
    # to our variable, r.
    char_len = 0
    for word in words:
        char_len += Ws["w" + str(word["index"])] * word2len[word["index"]]
        char_len += Ws["w" + str(word["index"])] * 1 # for spaces

    m.addConstr((char_len - 1) <= r) # -1 is needed here b/c there is one over counted space in above eq.
                                     # 4 tokens = 4 deps = 3 spaces, e.g. "I love to compute"
    m.addConstr(char_len >= 1) # model must answer something


    # equation 7
    # We have only one syntactic constraint which states that a subordinate
    # conjunction (sc) should be preserved if and only if the clause it belongs
    # to functions as a subordinate clause (sub) in the output.

    # this uses I think a old edge label "sub" which is not covertable to UD
    # The UD equivalent is that dependenents of edges

    #for m in mark_edges():
    #    gov, dep = m

    # I am not sure how to implement this # TODO

    # this is a debugging method. usually force_debug = []
    if len(force_debug) > 0:
        for edge in force_debug:
            a, b = edge
            dep = [_ for _ in jdoc["enhancedDependencies"] if _["governor"] in (a,b) and _["dependent"] in (a,b)][0]
            m.addConstr(Xs["{}-{}".format(dep["governor"], dep["dependent"])] >= 1) 
        for dep in jdoc["enhancedDependencies"]:
            e = (dep["governor"], dep["dependent"])
            if e not in force_debug:
                m.addConstr(Xs["{}-{}".format(dep["governor"], dep["dependent"])] <= 0)

    m.optimize()

    if m.status != GRB.Status.OPTIMAL:
        return {"compressed":"ILP not solved", "solved":False}
    else:
        '''
        this gets values of the ILPs X 'decision' vars
        note that I am using the .Xn function and the the .X
        .Xn gets the values of the Nth solution.
        http://www.gurobi.com/documentation/8.0/examples/poolsearch_py.html#subsubsection:poolsearch.py
        '''
        def get_Xs():
            for dep in jdoc["enhancedDependencies"]:
                if Xs['{}-{}'.format(dep["governor"], dep["dependent"])].Xn == 1:
                    yield {"word":dep["dependentGloss"], "dependent":dep["dependent"], "X":'{}-{}'.format(dep["governor"], dep["dependent"]),"X_val":Xs['{}-{}'.format(dep["governor"], dep["dependent"])].X}

        def get_compressed():
            indexes = list(set([int(_.replace("w","")) for _ in Ws if Ws[_].Xn == 1]))
            indexes.sort()
            compressed = " ".join([_["word"] for _ in words if _["index"] in indexes])
            return compressed

        def get_Ws():
            return [(Ws[_].Xn,_) for _ in Ws] 

        out = []
        for solution in range(0, m.SolCount):
            m.setParam(GRB.Param.SolutionNumber, solution)
            Xs_final = list(get_Xs())
            Xs_final.sort(key=lambda x:x["dependent"])
            def to_ints(key__):
                a,b = key__.split("-")
                return (int(a), int(b))
            predicted = [to_ints(key_) for key_, v in Xs.items() if v.Xn == 1]
            predicted.sort(key=lambda x:x[0])
            compressed = get_compressed()
            ws = get_Ws()
            out.append({"SolutionNumber":solution,
                        "objective_val": m.PoolObjVal,
                        "solved": True,
                        "ws": ws,
                        "get_Xs": Xs_final,
                        "get_Xs2": list(get_Xs_fix())
                        "predicted": predicted,
                        "compressed": compressed})
        if k == 1: # if k = 1 just return the best
            return out[0]
        else:
            return out # else return the k best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('-Q', action="store", dest="Q", type=str, required=False)

    parser.add_argument('-sent_no', action="store", dest="sent_no", type=int, required=True)

    parser.add_argument('-r', action="store", dest="r", help="the max char constraint", type=int, required=True)

    args = parser.parse_args()

    if args.Q is None:
        args.Q = []
    else:
        args.Q = args.Q.split()

    print("***")
    print(args)
    print("***")

    for lno, ln in enumerate(open("cache/ds.json", "r")):
        if lno == args.sent_no:
            jdoc = json.loads(ln)
            break

    print(" ".join([_["word"] for _ in jdoc["tokens"]]))
    output = run_model(jdoc, r=args.r, Q=args.Q)
    print(output)
