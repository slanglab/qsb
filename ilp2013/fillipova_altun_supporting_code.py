'''
This file holds helper methods for the structured perceptron
'''
from __future__ import division
from bottom_up_clean.utils import bfs
from unidecode import unidecode
import gzip
import string
import _pickle as pickle
import numpy as np
import json
import random
from code.log import logger
from scipy.sparse import csr_matrix
from itertools import product
from scipy.sparse import hstack
from scipy.sparse import find

PUNCT = [_ for _ in string.punctuation]
PUNCT.remove("$") # the dollar sign is a semantically meaningful token and needs to be included
PUNCT.remove("%")
PUNCT.append("''")
PUNCT.append('``')



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


def rand_weights(vocab):
    '''
    This is used for testing. It returns random weights.
    '''
    namer = FeatureNamer(vocabs=vocab)
    weights = np.random.rand( len(namer.lexical_names)+  len(namer.semantic_names) + len(namer.structural_names) + len(namer.syntactic_names))
    weights *= 50
    return weights


def zero_weights(vocab):
    '''
    This is used for initialization.  It returns a zero weight vector
    '''
    namer = FeatureNamer(vocabs=vocab)
    weights = np.zeros( len(namer.lexical_names)+  len(namer.semantic_names) + len(namer.structural_names) + len(namer.syntactic_names))
    return weights


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

def jdoc2goldlist(jdoc):
    '''return a pair of tuples, (gov, dep) for all in gold'''
    return [(_["governor"], _["dependent"]) for _ in jdoc["enhancedDependencies"]]


def A_but_not_B(A,B):
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
    i,j,v, = find(features) # get the coordinates and values of non-0 entries
    for j,v in zip(j,v):
        weights[j] -= v * epsilon

def add_features(features, weights, epsilon, t=None):
    '''
    features is a 1 X F sparse coo matrix
    weights is a numpy array
    this adds f[0][j]=v from weights[j] for all v > 0 in f
    '''
    i,j,v, = find(features) # get the coordinates and values of non-0 entries
    for j,v in zip(j,v):
        weights[j] += v * epsilon

def non_averaged_update(gold, predicted, w_t, vocabs, jdoc, epsilon=1):
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
    assert all(type(e) == tuple for e in gold)
    assert all(type(e) == tuple for e in predicted)
    for e in A_but_not_B(gold, predicted):
        features = f(e, jdoc, vocabs)
        add_features(features=features, weights=w_t, epsilon=epsilon)
    for e in A_but_not_B(predicted, gold):
        features = f(e, jdoc, vocabs)
        subtract_features(features=features, weights=w_t, epsilon=epsilon)
    return w_t

def get_tok(index, jdoc):
    '''
    Get the token dictionary at the index from CoreNLP dict
    A helper method
    '''
    if index == 0:
        return {"index":0, 'word':'ROOT', 'ner':'O',
                'pos': 'ROOT', 'lemma':'ROOT'}
    for _ in jdoc["tokens"]:
        if _["index"] == index:
            return _
    print(index,jdoc["tokens"])
    assert "unknown" == "token" 


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
            except IndexError: # in some cases it will not be possible to add an edge w/ the rules above. these might be revisited eventually.e.g. nmod:such_as, there is no token such_as.
                pass

    def unidecode_tok(t):
        t["word"] = unidecode(t["word"])
        return t
    jdoc["tokens"] = [unidecode_tok(t) for t in jdoc["tokens"]]
    jdoc["tokens"] = [_ for _ in jdoc["tokens"]]
    retained_indexes = [_["index"] for _ in jdoc["tokens"]]
    if "compression_indexes" in jdoc: # some test cases dont go thru preproc/proc_fillipova.py so they don't have this.
                                      # eventually it would be go to run them thru this for more uniformity but also prob not worth it
        jdoc["compression_indexes"] = list(set([_ for _ in jdoc["compression_indexes"] if _ in retained_indexes]))

    jdoc["enhancedDependencies"] = [_ for _ in jdoc['enhancedDependencies'] if _["dependentGloss"]]
    jdoc["enhancedDependencies"] = [_ for _ in jdoc['enhancedDependencies'] if _["governorGloss"]]

    return jdoc


def get_all_vocab_quick_for_tests():
    '''
    This is a dummy version of get_all_vocabs that goes very fast for testing
    '''
    with gzip.open("tests/fixtures/all_vocabs.json.p", "r") as inf:
        all_vocabs = pickle.load(inf)

    def product_as_strings(l1,l2):
        '''make the cross product of lists as strings, in the format A-B'''
        for a,b in product(l1, l2):
            yield "{}-{}".format(a,b)

    # these next two lines are slow but ultimately faster than precomuting and
    # loading from disk
    all_vocabs["lemma_v_dep_v"] = list(product_as_strings(all_vocabs["lemma_v"],
                                                          all_vocabs["dep_v"]))
    kys = set(all_vocabs.keys())
    for v in kys:
        v2n = {v:k for k,v in enumerate(all_vocabs[v])}
        all_vocabs.update({v +"2n": v2n})

    return all_vocabs


def get_all_vocabs():
    '''
    This gets the whole vocabulary for the whole corpus.
    See $fab preproc for more

    It takes about 30 seconds to load everything so I use
    get_all_vocab_quick_for_tests for testing
    '''
    with open("preproc/vocabs", "r") as inf:
        all_vocabs = json.load(inf)

    kys = set(all_vocabs.keys())
    for v in kys:
        v2n = {v:k for k,v in enumerate(all_vocabs[v])}
        all_vocabs.update({v +"2n": v2n})

    return all_vocabs


class FeatureNamer(object):
    '''
    This is a class that keeps track of the names of each feature in the vector
    for later analysis. The weights are a giant numpy vector so you need to
    map the float to the actual meaning. This class handles that mapping
    '''
    def add_prefix(self, prefix, L):
        '''add a prefix to every item on the list'''
        return [prefix + l for l in L]

    def __init__(self, vocabs):
        '''
        vs is a vocab dictionary, see get_all_vocabs
        '''
        self.vocabs = vocabs
        self.lexical_names = self.add_prefix("lemma_n:", vocabs["lemma_v"]) + self.add_prefix("siblings:", vocabs["lemma_v_dep_v"]) + self.add_prefix("lemma_n_label:", vocabs["lemma_v_dep_v"])
        self.syntactic_names = self.add_prefix("pos_h:", vocabs["pos_v"]) + self.add_prefix("pos_n:", vocabs["pos_v"]) + self.add_prefix("label_h_n:", vocabs["dep_v"])
        self.structural_names = ["depth_n", "num_children_n", "num_children_h", "char_length_n", "no_words_in"]
        self.semantic_names = self.add_prefix("ner_h:", vocabs["ner_v"]) + self.add_prefix("ner_n:", vocabs["ner_v"])  + ["negated"]


def get_random_dependency_scores(jdoc):
    '''
    This is a function for testing code.

    Instead of running a featurizer and taking the dot product w/ weights it
    just generates a random number
    '''
    return {"{}-{}".format(d["governor"], d["dependent"]): random.random() for
            d in jdoc["enhancedDependencies"]}


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


def get_siblings(e, jdoc):
    '''
    This gets the other children of h (that are not n). See below

    inputs:
        e(int,int):an edge from head, h, to node n
        jdoc(dict): a sentence from CoreNLP
    returns
        - other children of h that are not e
    '''
    h, n = e
    sibs = [i for i in jdoc["enhancedDependencies"] if i["governor"] == h and i["dependent"] != e]
    #sibs = list(filter(lambda x:x["governor"] == h and x["dependent"] != e,
    #            jdoc["enhancedDependencies"]))
    return [_['dependent'] for _ in sibs]


def get_edge(h, n, jdoc):
    '''A helper method: get edge between h and n'''
    out = [_ for _ in jdoc["enhancedDependencies"] if _["governor"] == h
          and _["dependent"] == n]
    return out.pop()
    #return filter(lambda x:x["governor"] == h and x["dependent"] == n,
    #              jdoc["enhancedDependencies"]).pop()


def label(h, n, jdoc):
    '''A helper method: return the label of dependency edge between h and n in jdoc'''
    return get_edge(h, n, jdoc)["dep"]


def get_children(index, jdoc, deps_kind = 'enhancedDependencies'):
    '''
    A helper method: get the children of a given vertex in a parse tree
    '''
    filtr = [_["dependent"] for _ in jdoc[deps_kind] if _["governor"] == index]
    #filtr = filter(lambda x:x["governor"] == index, jdoc[deps_kind])
    return filtr  #[_["dependent"] for _ in filtr]


def is_negated(n, jdoc):
    '''
    A helper method

    inputs:
        n(int): a token index
        jdoc(dict): a CoreNLP sentence dictionary
    returns:
        bool: is token n negated?
    '''
    child_ix = get_children(index=n, jdoc=jdoc)
    return any(label(n, cix, jdoc=jdoc) == "neg" for cix in child_ix)


def pos(index, jdoc):
    '''A helper method: returns pos tag for a token w/ index=index'''
    return get_tok(index, jdoc)["pos"]


def lemma(index, jdoc):
    '''A helper method: return the lemma for the token at input index'''
    return get_tok(index, jdoc)["lemma"]


def ne_tag(index, jdoc):
    '''A helper method: return the ne tag for the token at index'''
    return get_tok(index, jdoc)["ner"]


def num_children(index, jdoc):
    '''A helper method: return the number of children that a token has'''
    return len(get_children(index,jdoc))


def get_mark_edges(jdoc):
    '''A helper method: yield all mark edges'''
    for _ in jdoc["enhancedDependencies"]:
        if _["dep"] == "mark":
            yield (_["governor"], _["dependent"])


def char_length(index, jdoc):
    '''A helper method: return the char length of the token'''
    return len(get_tok(index, jdoc)["word"])


def no_words_in(index, jdoc):
    '''as far as I can tell this is the token index'''
    return get_tok(index, jdoc)["index"] # TODO


def depth(index, d):
    '''
    returns the depth of the index token in the parse tree
        inputs:
            index(int): an index in the parse tree
            d(dict): returned from bfs function in code.treeops
    '''
    return d[index]


def syntactic(e, jdoc, vocabs):
    '''
    return syntactic features in Filipova/Altun 2013
    inputs:
        e(tuple): h, n (see paper)
        jdoc(dict): stanford CoreNLP json sentence
        vocabs(dict): vocabs, i.e. all dep, all pos, all V etc.
    '''
    h, n = e

    row_size = len(vocabs["pos_v"]) * 2 + len(vocabs['dep_v'])

    try:
        row = np.array([0, 0, 0])
        col = np.array([vocabs['pos_v2n'][pos(h, jdoc)],
                        len(vocabs['pos_v2n']) + vocabs['pos_v2n'][pos(n, jdoc)],
                        len(vocabs['pos_v2n']) * 2 + vocabs["dep_v2n"][label(h, n, jdoc)]
                        ])
        data = np.array([1, 1, 1])

        sparse = csr_matrix((data, (row, col)), shape=(1, row_size), dtype=np.int8)

        return sparse
    except KeyError: #oov test
        row = np.array([0,0,0])
        col = np.array([0,0,0])
        return csr_matrix((np.array([0,0,0]),(row, col)), shape=(1,row_size), dtype=np.int8)
    

def semantic(e, jdoc, vocabs):
    '''
    return semantic features in Filipova/Altun 2013
    inputs:
        e(tuple): h, n (see paper)
        jdoc(dict): stanford CoreNLP json sentence
        vs(dict): vocabs, i.e. all dep, all pos, all V etc.
    '''

    h, n = e

    row_size = len(vocabs["ner_v"]) * 2 + 1

    row = np.array([0, 0, 0])
    col = np.array([vocabs['ner_v2n'][ne_tag(h, jdoc)],
                    len(vocabs['ner_v2n']) + vocabs['ner_v2n'][ne_tag(n, jdoc)],
                    len(vocabs['ner_v2n']) * 2
                    ])
    data = np.array([1, 1, is_negated(n, jdoc)])

    sparse = csr_matrix((data, (row, col)), shape=(1, row_size), dtype=np.int8)

    return sparse

def structural(e, jdoc):
    '''
    return structural features features in Filipova/Altun 2013
    inputs:
        e(tuple): h, n (see paper)
        jdoc(dict): stanford CoreNLP json sentence
    '''
    h, n = e
    d, pi, c = bfs(jdoc, hop_s=0)
    out = [depth(index=n, d=d),
           num_children(index=n, jdoc=jdoc),
           num_children(index=h, jdoc=jdoc),
           char_length(index=n, jdoc=jdoc),
           no_words_in(index=n, jdoc=jdoc)]

    return csr_matrix(np.asarray(out), dtype=np.int8)


def lexical(e, jdoc, vocabs):
    '''
    return lexical features features in Filipova/Altun 2013
    inputs:
        e(tuple): h, n (see paper)
        jdoc(dict): stanford CoreNLP json sentence
        vocabs(dict): these are created in $fab preproc
    '''

    h, n = e

    row_size = len(vocabs["lemma_v"]) + len(vocabs["lemma_v_dep_v"]) + len(vocabs["lemma_v_dep_v"])

    row = []
    col = []
    data = []

    try:
        col.append(vocabs["lemma_v2n"][lemma(n, jdoc)])
        row.append(0)
        data.append(1)
    except KeyError:
        pass # oov. test time

    for sib in get_siblings(e=e, jdoc=jdoc):
        turn_on = lemma(index=h, jdoc=jdoc) + "-" + label(h=h,n=sib,jdoc=jdoc)
        try: 
            col.append(len(vocabs["lemma_v2n"]) + vocabs["lemma_v_dep_v2n"][turn_on])
            row.append(0)
            data.append(1)
        except KeyError:
            pass # oov 
            

    turn_on = lemma(h, jdoc) + "-" + label(h,n,jdoc)

    try:
        col.append(len(vocabs["lemma_v2n"]) + len(vocabs["lemma_v_dep_v2n"]) + vocabs["lemma_v_dep_v2n"][turn_on])
        row.append(0)
        data.append(1)
    except KeyError:
        pass # oov, possible at test time

    sparse = csr_matrix((data, (row, col)), shape=(1, row_size), dtype=np.int8)

    return sparse

def f(e, jdoc, vocabs):
    '''

    This is the feature function from Fillipova & Altun 2013

    input:
        e(tuple): an edge in the parse tree
        jdoc(dict): a json sentence from CoreNLP
        vocabs(dict): a dictionary of vocabularies like
                         lemma vocab, pos vocab, etc.
    returns:
        a feature vector (np.ndarray)
    '''
    feats = [syntactic(e, jdoc, vocabs),
             semantic(e, jdoc, vocabs),
             structural(e, jdoc),
             lexical(e, jdoc, vocabs)]
    return hstack(feats)

def get_featurized_dependency_scores(jdoc, vs, weights):
    '''
    This returns w(e) for each edge in jdoc.

    This is equation 6 in Fillipova & Altun 2013
    '''
    def to_edge(d):
        return (d["governor"], d["dependent"])
    out = {"{}-{}".format(d["governor"], d["dependent"]): f(e=to_edge(d), jdoc=jdoc, vocabs=vs).dot(weights.T)[0] for d in jdoc["enhancedDependencies"]}
    out = {k:float(v) for k,v in out.items()}
    return out
