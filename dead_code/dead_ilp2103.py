
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
        pass  # oov, possible at test time

    sparse = csr_matrix((data, (row, col)), shape=(1, row_size), dtype=np.int8)

    return sparse


def get_all_vocab_quick_for_tests():
    '''
    This is a dummy version of get_all_vocabs that goes very fast for testing
    '''
    with gzip.open("tests/fixtures/all_vocabs.json.p", "r") as inf:
        all_vocabs = pickle.load(inf)

    def product_as_strings(l1, l2):
        '''make the cross product of lists as strings, in the format A-B'''
        for a, b in product(l1, l2):
            yield "{}-{}".format(a, b)

    # these next two lines are slow but ultimately faster than precomuting and
    # loading from disk
    all_vocabs["lemma_v_dep_v"] = list(product_as_strings(all_vocabs["lemma_v"],
                                                          all_vocabs["dep_v"]))
    kys = set(all_vocabs.keys())
    for v in kys:
        v2n = {v: k for k, v in enumerate(all_vocabs[v])}
        all_vocabs.update({v + "2n": v2n})

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
        v2n = {v: k for k, v in enumerate(all_vocabs[v])}
        all_vocabs.update({v + "2n": v2n})

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
    sibs = [i for i in jdoc["enhancedDependencies"] if i["governor"] == h and i["dependent"] != n]
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


def get_tok(index, jdoc):
    '''
    Get the token dictionary at the index from CoreNLP dict
    A helper method
    '''
    if index == 0:
        return {"index": 0, 'word': 'ROOT', 'ner': 'O',
                'pos': 'ROOT', 'lemma': 'ROOT'}
    for _ in jdoc["tokens"]:
        if _["index"] == index:
            return _
    print(index, jdoc["tokens"])
    assert "unknown" == "token"