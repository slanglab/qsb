#!/usr/bin/python
'''
Global Inference for Sentence Compression:
An Integer Linear Programming Approach

James Clarke and Mirella Lapata, JAIR 2008

This implements the LM and LM+Constr models from the paper, with the addition
of a query term, Q=[w1, w2, w3...], a list of words.
   -  To run LM+Constr set grammar=True

We use Gurobi for the ILP

We use universal dependencies to enforce the grammatical constraints, instead
of the RASP parser (a constituent parser). Translating the constrints into UD
was very straight forward. We make this choice because UD is a standard
representation for modern NLP.
'''
import sys
import json
import argparse
from gurobipy import *
from ilp.supporting_code import *

def run_model(jdoc, min_toks=1, r=100000, grammar=False, Q=[], verbose=False):
    '''
    Run the Clarke and Lapata 2008 model with Gurobi

    inputs:
        jdoc(dict): a single sentence jdoc dictionary from CoreNLP
        r(int): max character constraint
        min_toks(int): how many required tokens
        grammar(bool): use the grammatical constraints, LM + Constr from paper
        Q(list): a list of strings which must be included.
                    - one corner case is that a word is included twice in a sentence
                      e.g. The dog is a dog
                      In this case, only the first word will be included
    '''

    words = [_["word"] for _ in jdoc["tokens"]]

    # these assertions to not apply to transformed trees
    #assert min([_["index"] for _ in jdoc["tokens"]]) == 1
    #assert max([_["index"] for _ in jdoc["tokens"]]) == len(jdoc["tokens"])

    if verbose:
        print Q

    # these from kenlm
    alpha_scores = get_alpha_scores(words)
    gamma_scores = get_gamma_scores(words)
    beta_scores = get_beta_scores(words)

    # Model
    m = Model("clarke_lapata_2008")

    if not verbose:
        m.setParam('OutputFlag', 0)

    '''init the alpha, beta and delta variables'''

    # note the +1 indexing. Following the JAIR notation exactly here.
    # alpha[0] does not exist. the 0th token is the special token x0
    alpha = {}
    for k, w in enumerate(words):
        alpha[k + 1] = m.addVar(name="alpha:" + w, vtype=GRB.BINARY)

    beta = generate_beta_indexes(words)
    for i in beta:
        for j in beta[i]:
            beta[i][j] = m.addVar(name=beta[i][j], vtype=GRB.BINARY)

    delta = {}
    for k, w in enumerate(words):
        delta[k + 1] = m.addVar(name=w, vtype=GRB.BINARY)

    gamma = enum_gamma(words)
    for i in gamma:
        for j in gamma[i]:
            for k in gamma[i][j]:
                assert j > 0 and k > 0
                gamma[i][j][k] = m.addVar(name=gamma[i][j][k], vtype=GRB.BINARY)

    # these are just lists of all indexes into beta and gamma variables 
    beta_indexes = [(i,j) for i in beta_scores for j in beta_scores[i]]
    gamma_indexes = [(i,j,k) for i in gamma_scores for j in gamma_scores[i] for k in gamma_scores[i][j]]

    m.setObjective(sum(alpha[k + 1]*alpha_scores[f] for k,f in enumerate(words)) +
                   sum(beta[i][j]*beta_scores[i][j] for i,j in beta_indexes) +
                   sum(gamma[i][j][k] * gamma_scores[i][j][k] for i,j,k in gamma_indexes),
                   GRB.MAXIMIZE)

    betaindex2string = generate_beta_indexes(words)
    gammaindex2string = enum_gamma(words)
    word2len = get_word_to_len(words)

    # contstraint 1, only one word can start the sentence
    m.addConstr(sum([alpha[k + 1] for k,f in enumerate(words)]) == 1)

    # constraint 2
    for k in range(1, len(words) + 1):
        total = delta[k] # delta indexed at 1
        total -= alpha[k]
        for i in range(0, k-1): # each of these ks are +1 b/c  python range indexing, as compared to eq. 5
            for j in range(i + 1, k):
                total -= gamma[i][j][k]
        m.addConstr(total == 0)

    # constraint 3 If a word is included in the sentence it must either be preceded by one
    # word and followed by another or it must be preceded by one word and end the sentence.
    for j in range(1, len(words) + 1):
        total = delta[j]
        for i in range(0, j): # not j-1 cuz python indexing
            for k in range(j + 1, len(words) + 1): # +1 b/c python indexing
                total -= gamma[i][j][k]
        for i in range(0, j):
            total -= beta[i][j]
        m.addConstr(total == 0)

    # constraint 4
    for i in range(1, len(words) + 1):
        total = delta[i]
        # If a word is in the sentence it must be followed by two words
        for j in (i + 1, len(words)): # n - 1 = len(words) b/c of python loops
            for k in range(j + 1, len(words) + 1):
                total -= gamma[i][j][k]
        # or followed by one word and then the end of the sentence
        for j in range(i+1, len(words) + 1):
            total -= beta[i][j]
        # or it must be preceded by one word and end the sentence.
        for h in range(0, i): # add 1 to i vs. notation in JAIR b/c of python indexing
            total -= beta[h][i]
        m.addConstr(total == 0)

    # constraint 5, only 1 bigram can end the sentence
    m.addConstr(sum([beta[i][j] for i,j in beta_indexes]) == 1)

    # AH: query constraint
    for word_ in Q:
        m.addConstr(delta[words.index(word_) + 1] == 1)

    # min tok constraint. needed for lang model b/c the score values are negative
    # so the model can maximize objective by shedding tokens :)
    # AFICT the only way the model will go over this is b/c of grammatical
    # constrains
    m.addConstr(sum([delta[i] for i in delta]) >=  min_toks)

    char_len = 0
    for k, w in enumerate(words):
        char_len += delta[k + 1] * word2len[w]
        char_len += 1 # for the space

    m.addConstr(char_len <= r)   

    def ilp_if_statement(A, B):
        '''if a --> b'''
        m.addConstr(B - A >= 0)

    def ilp_iff_statement(A, B):
        '''if a <--> b'''
        m.addConstr(B - A == 0)

    def iff_cross_product(group):
        '''
        for (i,j) \in L X L, i <--> j

        e.g. for [1,2,3], assert 1 <--> 2, 1 <--> 3, 2 <--> 3
        '''
        for i,j in itertools.combinations(group, 2):
            ilp_iff_statement(delta[i], delta[j])

    def diff(i,j):
        return delta[i] - delta[j]

    if grammar:

        # eq (20)
        nc_mods = get_ncmods(jdoc) # UD conversion of ncmods
        for n in nc_mods: # for all non clausal modifiers
            for c in n["children"]: # for all children...
                # if the child is included, then governor is too
                ilp_if_statement(delta[c], delta[n["governor"]])
                # not that 'children' includes the dependent of ncmod relation

        # eq (21) determiners
        determiners = get_det_mods(jdoc)
        for dep, gov in determiners:
            # dep will be the determiner. #TODO test/check
            ilp_if_statement(delta[dep], delta[gov])

        # eq. (22) forces negations
        negations = get_negations(jdoc)
        for group in negations:
            iff_cross_product(group)

        # eq. (23) forces possessives
        for group in get_possessives(jdoc):
            iff_cross_product(group)

        # eq. (24) forces verb arguments
        verb_groups = get_verb_groups(jdoc)
        for group in verb_groups:
            verb = group['verb']
            for arg in group["args"]:
                ilp_iff_statement(delta[verb], delta[arg])

        # eq. (25) forces the compression to contain at least one verb
        # no +1 b/c corenlp indexes at 1
        m.addConstr(sum([delta[i] for i in get_verb_ix(jdoc)]) >=  1)

        # eq. (26) & eq. (27)
        pps = get_PPs(jdoc)
        for prep_phrase in pps:
            children = prep_phrase["children"]
            introducing_preposition = prep_phrase["introducing_preposition"]
            for child in children:
                # if the child is included, introducing_preposition must be too
                ilp_if_statement(delta[child],delta[introducing_preposition])

            sum_of_diffs = sum([diff(c,introducing_preposition) for c in children])
            m.addConstr(sum_of_diffs >= 0)

        sbars = get_SBARs(jdoc)
        for sbar in sbars:
            children = sbar["children"]
            introducing_word = sbar["introducing_word"]
            for child in children:
                # if the child is included, introducing_word must be too
                ilp_if_statement(delta[child],delta[introducing_word])

            sum_of_diffs = sum([diff(c,introducing_word) for c in children])
            m.addConstr(sum_of_diffs >= 0)

        # eq. (32) forces the compression to contain personal pronouns
        pronouns = get_personal_pronouns(jdoc)
        for p in pronouns:
            m.addConstr(delta[p] == 1)

        coordinated = get_coordination(jdoc)
        for coord in coordinated:
            i = coord["coordinator"]
            j = coord["conjunct1"]
            k = coord["conjunct2"]
            m.addConstr((1-delta[i]) + delta[j] >= 1) #eq 28
            m.addConstr((1-delta[i]) + delta[k] >= 1) #eq 29
            m.addConstr((delta[i] + (1 - delta[j])  + (1 - delta[k])) >= 1) #eq 30

    m.optimize()

    '''these functions get the final values of the ILP 'decision' vars'''
    def get_alphas():
        for k, f in enumerate(words):
            if alpha[k + 1].X > 0:
                yield({"word":f, "alpha_val":alpha[k + 1], "k":k + 1})

    def get_betas():
        for i,j in beta_indexes:
            if beta[i][j].X > 0:
                yield({"str": betaindex2string[i][j], "beta_val":beta[i][j].X, "i": i, "j":j})

    def get_gammas():
        for i,j,k in gamma_indexes:
            if gamma[i][j][k].X > 0:
                yield({"str":gammaindex2string[i][j][k],"gamma_val":gamma[i][j][k], "i":i,"j":j,"k":k})

    def get_deltas():
        for k, f in enumerate(words):
            if delta[k + 1].X > 0:
                yield({"word":f, "delta_val":delta[k + 1], "k":k + 1})


    if m.status == GRB.Status.OPTIMAL:
        objective_val = m.objVal
    else:
        objective_val = None

    def get_compression():
        '''Return the compression from the ILP'''
        if m.status != GRB.Status.OPTIMAL:
            return "ILP not solved"
        deltas = sorted([_ for _ in get_deltas() if _["delta_val"] > 0], key=lambda x:x["k"])

        deltas = [_["k"] for _ in get_deltas()]

        words = [_['word'] for _ in jdoc["tokens"] if _["index"] in deltas]

        return " ".join(words)

    if m.status != GRB.Status.OPTIMAL:
        return {"compressed":"ILP not solved", "solved":False}
    else: 
        return {"objective_val": objective_val,
                "solved": m.status == GRB.Status.OPTIMAL,
                "alphas": list(get_alphas()),
                "betas": list(get_betas()),
                "gammas": list(get_gammas()),
                "deltas": list(get_deltas()),
                "compressed": get_compression()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('-grammar', action="store_true", default=False)

    parser.add_argument('-toks', action="store", dest="toks", type=int, required=True)

    parser.add_argument('-Q', action="store", dest="Q", type=str, required=False)

    parser.add_argument('-sent_no', action="store", dest="sentence_no", type=int, required=True)

    args = parser.parse_args()

    if args.Q is None:
        args.Q = []
    else:
        args.Q = args.Q.split()

    print "***"
    print args
    print "***"

    MIN_TOKS = args.toks

    assert MIN_TOKS is not None

    SENTENCE_NO = args.sentence_no

    cache = [_.replace("\n", "") for _ in open("cache/ds.json")]

    jdoc = json.loads(cache[SENTENCE_NO])

    output = run_model(jdoc, min_toks=MIN_TOKS, grammar=False, Q=args.Q)

    included = filter(lambda x:x["delta_val"] > 0, output["deltas"])

    print map(lambda x:x["word"], included)
