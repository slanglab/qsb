import itertools
import json
from collections import defaultdict
from code.treeops import dfs
import kenlm as klm

from configs import klm


def load_sentence(filename):
    with open(filename, "r") as inf:
        jdoc = json.load(inf)["sentences"][0]
        return jdoc

def default_to_regular(d): #[1]
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def enum_alpha(_words):
    '''enumerate the alpha variables in the cohen/lapata model'''
    assert all(type(_) in [str] for _ in _words), "words should be a list of str"
    n = len(_words) + 1
    return {i:_words[i-1] for i in range(1, n)}

def get_word_to_len(_words):
    '''return word -> len(word) for word in words'''
    return {w:len(w) for w in _words}

def get_ncmods(jdoc_):
    '''
    Clarke and Lapata say..

    "if we include a non-clausal modifier (ncmod) in the compression
    (such as an adjective or a noun) then the head of the modifier must
    also be included."

    They use the RASP parser. The RASP docs say
    "ncmod encodes binary relations between non-clausal modifiers and their heads"
    https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-662.pdf

    I think these correspond to the UD relations: amod, advmod, nummod, nmod, appos (?)
    '''
    rels = ["amod", "advmod", "nmod", "nummod", "appos"]
    relations = [_ for _ in jdoc_["basicDependencies"] if _["dep"] in rels]
    for r in relations:
        dependent_and_children_indexes = dfs(g=jdoc_, hop_s=r["dependent"], D=[])
        yield {'governor': r["governor"],
              'children':dependent_and_children_indexes}

def generate_beta_indexes(_words):
    '''
    enumerate the beta variables in the cohen/lapata model
    inputs:
        _words(list): list of string unigrams
        _out(dict of dicts): (int: i,int: j) -> word[i] + word[j]
    '''
    n = len(_words)
    _out = defaultdict(lambda: defaultdict())
    alphas = enum_alpha(_words)

    for i in range(0, n):
        for j in range(i + 1, n + 1):
            first = alphas[i] if i > 0 else "x0" # if i = 0, first token is x0
            _out[i][j] = first + "-" + alphas[j]
    return default_to_regular(_out)

def enum_gamma(_words):
    '''enumerate the gamma variables in the cohen/lapata model'''
    n = len(_words)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    alphas = enum_alpha(_words)

    for i in range(0, n - 1):
        for j in range(i + 1, n):
            for k in range(j + 1, n + 1):
                first = alphas[i] if i > 0 else "x0"
                out[i][j][k] = first + "-" + alphas[j] + "-" + alphas[k]

    return default_to_regular(out)

def get_beta_scores(_words):
    betas = generate_beta_indexes(_words)
    out = defaultdict(lambda: defaultdict())
    for i in betas:
        for j in betas[i]:
            str_ = betas[i][j].replace("-", " ").replace("x0", "SOS") + " EOS"
            out[i][j] = klm.score(str_)
    return default_to_regular(out)

def get_gamma_scores(_words):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    gammas = enum_gamma(_words)
    for i in gammas:
        for j in gammas[i]:
            for k in gammas[i][j]:
                str_ = gammas[i][j][k].replace("-", " ").replace("x0", "SOS")
                out[i][j][k] = klm.score(str_)
    return out

def get_alpha_scores(_words):
    '''take a list of words and return importance scores for those words'''
    assert type(_words) == list
    assert all(type(i) in [str] for i in _words)
    out = dict()
    for w in _words:
        out[w] = klm.score("SOS " + w)
    return out


def get_verb_ix(jdoc):
    '''
    get token indexes of verbs in the jdoc
        - these should align with the token ids of the alpha vars b/c corenlp
          indexes tokens at 1
    '''
    return [i["index"] for i in jdoc["tokens"] if i["pos"][0].lower() == "v"]


def get_coordination(jdoc):
    '''
    # We also wish to handle coordination. If two head words are conjoined in the source
    # sentence, then if they are included in the compression the coordinating conjunction must
    # also be included:
    '''
    words_conjoined = [(_["dependent"], _["governor"])
                        for _ in jdoc["basicDependencies"] if _["dep"] == "conj"]
    for w in words_conjoined:
        dep, gov = w
        children_of_gov = [_ for _ in jdoc["basicDependencies"] if _["governor"] == gov]
        for c in children_of_gov:
            if c["dep"] == "cc":
                yield {"coordinator":gov,
                       "conjunct1":dep,
                       "conjunct2":c["dependent"]} # these must be included

def get_personal_pronouns(jdoc):
    '''
    "Equation (32) forces personal pronouns to be included in the compression"
    '''
    return [i["index"] for i in jdoc["tokens"] if i["pos"].lower() == "prp"]

def get_det_mods(jdoc):
    '''
    if we include a determiner in the compression
    then the head of the determiner must also be included
    '''
    return [(_["dependent"], _["governor"]) for _ in jdoc["basicDependencies"] if _["dep"] == "det"]


def get_negations(jdoc):
    '''
    if we include a negation in the compression you need to include the head,
    so these pairs have to be included together
    '''
    negations = [(_["dependent"], _["governor"]) for _ in jdoc["basicDependencies"] if _["dep"] == "neg"]
    for dep, gov in negations:
        dep_tok = [i for i in jdoc["tokens"] if i["index"] == dep][0]
        gov_tok = [i for i in jdoc["tokens"] if i["index"] == gov][0]
        child_tok = None
        def is_verb(tok):
            return tok["pos"][0].lower() == "v"
        if is_verb(gov_tok):
            # AH: can you assume there is always a child tok? will get a runtime error if no on the
            # next 2 lines.
            chi = [_["dependent"] for _ in jdoc["basicDependencies"] if _["dep"] == "aux"][0]
           
            child_tok = [i for i in jdoc["tokens"] if i['index'] == chi][0] 
        if child_tok is not None:
            yield [dep_tok["index"], gov_tok["index"], child_tok["index"]]
        else:
            yield [dep_tok["index"], gov_tok["index"]]


def get_possessives(jdoc):
    '''
    if we include a negation in the compression you need to include the head,
    so these pairs have to be included together
    '''
    pairs = [(_["dependent"], _["governor"]) for _ in jdoc["basicDependencies"] if _["dep"] == "nmod:poss"]
    for p in pairs:
        dep, gov = p
        # e.g. Marie's book
        case_child = [_["dependent"] for _ in jdoc["basicDependencies"] if
                      _["governor"] == dep and _["dep"] == "case"]
        case_child_tok = filter(lambda x:x["index"] in case_child, jdoc["tokens"])
        for c in case_child_tok:
            assert c["word"].lower() == "'s" or c["word"].lower() == "'"
        case_child_tok = [_["index"] for _ in case_child_tok]
        case_child_tok.append(dep)
        case_child_tok.append(gov)
        yield set(case_child_tok)

def get_case(jdoc):
    '''
    case edges (e.g. possessives and prepositions) must be retained
    '''
    return [(_["dependent"], _["governor"]) for _ in jdoc["basicDependencies"] if _["dep"] == "case"]


def get_verb_groups(jdoc):
    '''
    Clarke and Lapata:
    "We thus force the program to make the same decision on the verb,
    its subject, and object.""
    '''
    verbs = get_verb_ix(jdoc)
    # core nominal arguments in UD
    rels = ['iobj', 'dobj', 'obj', 'nsubj']
    for v in verbs:
        group = [_["dependent"] for _ in jdoc["basicDependencies"] if
                 _["governor"] == v and _["dep"] in rels]
        yield {"verb":v, "args":group}

def to_lists(list_, out = []):
    '''return all nested lists in a list of lists (i.e. constituent parse)'''
    for i in list_:
        if type(i) == list and i not in out:
            out.append(i)
            to_lists(i, out)
    return out

def get_PPs(jdoc):
    '''return all nested PPs and sbars'''
    case_edges = [_ for _ in jdoc["basicDependencies"] if _["dep"] == "case"]

    # not all case edges are PPs

    def is_preposition(ix):
        pos = [_["pos"] for _ in jdoc["tokens"] if _["index"] == ix][0]
        return pos in ["IN", "TO"] # preposition + to

    pp_edges = [(_["governor"], _["dependent"])  for _ in case_edges if is_preposition(_["dependent"])]

    out = []
    indexes = [_["index"] for _ in jdoc["tokens"]]
    for gov, dep in pp_edges:
        children_and_v = dfs(g=jdoc, hop_s=gov, D=[])
        assert all(i in indexes for i in children_and_v)
        children_and_v.remove(gov)
        out.append({"children": children_and_v,
                    "introducing_preposition": dep})
    return out


def get_SBARs(jdoc):
    '''return all nested PPs and sbars'''
    mark_edges = [(_["governor"], _["dependent"]) for _ in jdoc["basicDependencies"] if _["dep"] == "mark"]

    for gov, dep in mark_edges:
        children = dfs(g=jdoc, hop_s=gov, D=[])
        children.remove(dep)
        yield {"children": children,
               "introducing_word": dep}
