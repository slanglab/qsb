'''
This file holds helper methods for the structured perceptron
'''
from __future__ import division
from unidecode import unidecode

from code.log import logger
from scipy.sparse import find
from bottom_up_clean.all import get_features_of_dep



def get_gold_y(jdoc):
    return [o["index"] in jdoc["compression_indexes"] for o in jdoc["tokens"]]


def get_pred_y(predicted_compression, original_indexes):
    assert all(type(o) == int for o in predicted_compression)
    return [o in predicted_compression for o in original_indexes]


def get_oracle_r(source_jdoc):
    '''
    Get the oracle compression length, in chars
    Note, F & A 2013: "The maximum permitted compression length is set to be the same as the length
    of the oracle compression."

    inputs:
        source_jdoc(dict): a sentence from CoreNLP
    returns:
        r(int): the compression rate, r
    '''
    compression_toks = [_ for _ in source_jdoc["tokens"] if _["index"] in source_jdoc["compression_indexes"]]
    r = len(compression_toks) - 1 # spaces
    r += sum([len(_["word"]) for _ in compression_toks])

    return r


def get_gold_edges(source_jdoc):
    '''
    This returns the gold edges in a compression.
    '''
    toks_in_source_in_compression = source_jdoc["compression_indexes"]
    toks_in_source_in_compression.append(0) # need the 0 to get ROOT to tok edges
    gold = [(_["governor"], _["dependent"]) for _ in source_jdoc["enhancedDependencies"] if _["governor"] in toks_in_source_in_compression and _["dependent"] in toks_in_source_in_compression]
    root_edges = [_ for _ in gold if _[0] == 0] # i.e. edges from root

    # This helper method is needed to find tokens that are governed by tokens
    # that are not root. The tree transform adds an edge to each of the
    # tokens in the sentence. Some of edges are not really gold edges but
    # they match the "gold" criteria above. So you need to pull them out in code
    # below.
    def has_non_root_governors(L, dep):
        '''does this token have non root governors'''
        non_root_gov = sum(1 for i in L if i[1] == dep and i[0] != 0)
        return non_root_gov > 0

    for edge in root_edges:
        root, dep = edge
        assert root == 0
        if has_non_root_governors(gold, dep):
            gold.remove(edge) # pluck out extraneous root-to-token edges
    assert set([0] + [_ for g in gold for _ in g]) == set(source_jdoc["compression_indexes"])
    return gold


def edge_precision(predicted, gold):
    '''returns the edge-level precision'''
    true_pos = sum(1 for i in predicted if i in gold)
    false_pos = sum(1 for i in predicted if i not in gold)
    p = true_pos/(true_pos + false_pos)
    assert p >= 0 and p <= 1
    return p


def edge_recall(predicted, gold):
    '''returns the edge-level recall'''
    true_pos = sum(1 for i in predicted if i in gold)
    false_negative = sum(1 for i in gold if i not in predicted)
    r = true_pos/(true_pos + false_negative)
    assert r >= 0 and r <= 1
    return r


def f1(predicted, gold):
    '''calculate f1 score'''
    if len(predicted) == 0:
        prec = 1.0
        rec = 0.0
    elif len(gold) == 0:
        logger.warning("gold is zero") # this happens on occasion b/c of parse differences I think. gold is EDGE-level
        return 0
    else:
        rec = edge_recall(predicted, gold)
        prec = edge_precision(predicted, gold)
    if (prec + rec) == 0:
        return 0
    return (2 * (prec * rec))/(prec + rec)


def A_but_not_B(A, B):
    '''
    return everything in A that is not in B
    inputs:
        A(list): a list
        B(list): another list
    '''
    return [_ for _ in A if _ not in B]


def subtract_features(features, weights, epsilon, t=None):
    '''
    features is a 1 X F sparse coo matrix
    weights is a numpy array
    this subtracts f[0][j]=v from weights[j] for all v > 0 in f
    '''
    i, j, v, = find(features) # get the coordinates and values of non-0 entries
    for j, v in zip(j, v):
        weights[j] -= v * epsilon


def add_features(features, weights, epsilon, t=None):
    '''
    features is a 1 X F sparse coo matrix
    weights is a numpy array
    this adds f[0][j]=v from weights[j] for all v > 0 in f
    '''
    i, j, v, = find(features)  # get the coordinates and values of non-0 entries
    for j, v in zip(j, v):
        weights[j] += v * epsilon


def non_averaged_update(gold, predicted, w_t, jdoc, vectorizer, epsilon=1):
    '''
    input:
        gold(list:tuple): a list of edges in gold, e.g. [(4,5), (5,6), (1,4)]
        gold(list:tuple): a list of edges in predicted, e.g. [(4,5), (5,6)]
        w_t(ndarray): the weights at time t
        vocabs(dict): the vocabulary
        jdoc(dict): a CoreNLP sentence
        epsilon: a learning rate
    returns:
        w_(t + 1): an updated weight vector w/ no averaging
    ''' 
    for e in A_but_not_B(gold, predicted): 
        features = f(e, jdoc, vectorizer)
        add_features(features=features, weights=w_t, epsilon=epsilon)
    for e in A_but_not_B(predicted, gold):
        features = f(e, jdoc, vectorizer)
        subtract_features(features=features, weights=w_t, epsilon=epsilon)
    return w_t


def filippova_tree_transform(jdoc):
    '''
    This is a UD version of the fillipova tree transform from
        - Filippova/Strube 2008
        - Filippova/Altun 2013
    '''
    def is_verb(tok):
        return tok["pos"] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    for v in jdoc["tokens"]:
        total_edges_from_root_to_this_v = sum(1 for i in jdoc["enhancedDependencies"] if i['governor'] == 0 and i["dependent"] == v["index"])
        if total_edges_from_root_to_this_v == 0:
            jdoc["enhancedDependencies"].append({"governor": 0, "dependent": v["index"],
                                                 "governorGloss": "ROOT", "dependentGloss": v["word"],
                                                 "dep": "ROOT"})
    
    
    '''
    This seems buggy and also not really true to our description of reimplementation in the appendix

    enhanced_nominal_modifiers = [_ for _ in jdoc["enhancedDependencies"] if "nmod:" in _['dep']]
    # see issue #10 on github
    for enhanced_edge in enhanced_nominal_modifiers:
        relevant_preposition = enhanced_edge["dep"].split(":").pop()
        dependent_of_enhanced = enhanced_edge["dependent"]
        children_of_dependent = [_["dependent"] for _ in jdoc["enhancedDependencies"] if _["governor"] == dependent_of_enhanced]
        children_of_dep_as_tokens = [_ for _ in jdoc["tokens"] if _["index"] in children_of_dependent]
        if not any(i["lemma"] == relevant_preposition for i in children_of_dep_as_tokens) and relevant_preposition not in ["tmod", "poss", "agent", "according_to", "npmod", "because_of"]:
            try:
                tokens_with_this_preposition = [_ for _ in jdoc["tokens"] if str(_["word"]).lower() == str(relevant_preposition).lower()]
                # find the closest preposition (heuristic)
                tokens_with_this_preposition.sort(key=lambda x:abs(x["index"] - enhanced_edge["dependent"]))
                token_with_this_preposition = tokens_with_this_preposition[0]
                jdoc["enhancedDependencies"].append({"governor": dependent_of_enhanced, "dependent": token_with_this_preposition["index"],"governorGloss": enhanced_edge["dependentGloss"], "dependentGloss": token_with_this_preposition["word"], "dep": "case"})
            except IndexError: # in some cases it will not be possible to add an edge w/ the rules above.
                pass
    '''

    def unidecode_tok(t):
        t["word"] = unidecode(t["word"])
        return t
    jdoc["tokens"] = [unidecode_tok(t) for t in jdoc["tokens"]]

    return jdoc


def f(e, jdoc, vectorizer):
    '''

    This is the feature function from Fillipova & Altun 2013

    input:
        e(tuple): an edge in the parse tree
        jdoc(dict): a json sentence from CoreNLP
        vectorizer(dict): sklearn dict vectorizer.
    returns:
        a feature vector (np.ndarray)
    '''
    return vectorizer.transform(get_features_of_dep(e, jdoc))


def get_featurized_dependency_scores(jdoc, weights, vectorizer):
    '''
    This returns w(e) for each edge in jdoc.

    This is equation 6 in Fillipova & Altun 2013
    '''
    def to_edge(d):
        return (d["governor"], d["dependent"])
    out = {"{}-{}".format(d["governor"], d["dependent"]): f(e=d, jdoc=jdoc, vectorizer=vectorizer).dot(weights.T)[0] for d in jdoc["enhancedDependencies"]}
    out = {k:float(v) for k,v in out.items()}
    return out


def get_q_word_and_governor(word_, jdoc):
    '''
    This gets the edge from a governor to a word. This is used to
    find edges which must be included so as to include the query, Q

    inputs:
        word_(int): a word index in the sentence
    returns:
        e(tuple): the index of the relation from the word's head to
                  the word in the jdoc
    '''

    assert word_ in (i["index"] for i in jdoc["tokens"])

    out = [_ for _ in jdoc["enhancedDependencies"]
           if _["dependent"] == word_]
    out.sort(key=lambda x:x["governor"]) # prefer root, if possible
    out = out[0]

    return '{}-{}'.format(out["governor"], out["dependent"])