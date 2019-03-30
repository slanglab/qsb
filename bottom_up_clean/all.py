import json
import string
import numpy as np
import copy
import random

from bottom_up_clean.utils import bfs
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer

NULLSET = set()

BLANK = {"lemma": "is_root", "word": "", "len": 0,  "index": 0, "ner": "O", "pos": "root"}

def get_siblings(e, jdoc):
    '''
    This gets the other children of h (that are not n). See below

    inputs:
        e(int,int):an edge from head, h, to node n. e[ix] is head 2 tail
        jdoc(dict): a sentence from CoreNLP
    returns
        - other children of h that are not e
    '''
    return {i["dependentGloss"] for i in jdoc["vx2children"][e[0]] if i["dependent"] != e[1]}


def get_features_of_dep(dep, sentence):
    '''
    Dep is some dependent in basicDependencies

    references to h and n come from F & A 2013 table 1
    '''

    depentent_token = sentence["ix2tok"][dep["dependent"]]

    try:
        governor_token = sentence["ix2tok"][dep["governor"]]
    except:
        governor_token = BLANK

    sibs = get_siblings(e=(dep["governor"], dep["dependent"]), jdoc=sentence)

    out = dict()

    ### These are the F and A (2013) features

    # syntactic
    out["dep"] = dep["dep"]
    for s in sibs:
        out["ds", s] = 1
    out["pos_h"] = governor_token["pos"]
    out["pos_n"] = depentent_token["pos"]

    # Structural
    out["childrenCount_h"] = sentence["childrencount"][dep["governor"]]
    out["childrenCount_e"] = sentence["childrencount"][dep["dependent"]]
    out["depth_h"] = sentence['depths'][dep["dependent"]]
    out["char_len_h"] = governor_token["len"]
    out["no_words_in_h"] = governor_token["index"]

    # Semantic
    out["ner_h"] = governor_token["ner"]
    out["ner_n"] = depentent_token["ner"]
    out["is_neg_n"] = dep["dep"] == "neg"

    # Lexical
    out['lemma_n'] = depentent_token["lemma"]
    out["lemma_h_label_e"] = governor_token["lemma"] + dep["dep"]
    for s in sibs:
        out["s:", s, governor_token["lemma"]] = 1

    return out


def len_current_compression(current_compression, sentence):
    '''get the character length of the current compression'''
    return sum(sentence['ix2tok'][o]["len"] for o in current_compression) + len(current_compression) - 1


def pick_l2r_connected(frontier, current_compression, sentence):
    connected = get_connected2(sentence, frontier, current_compression)
    
    # min index gets the first element, in L to R order
    if len(connected) > 0:
        return min(connected)
    else:
        return min(frontier)


def get_depths(sentence):
    '''just a wrapper for bfs that only returns the depths lookup'''
    depths, heads_ignored_var, c_ignored_var = bfs(g=sentence, hop_s=0)
    return depths


def init_frontier(sentence, Q):
    '''initalize the frontier for additive compression'''
    return sentence["indexes"].difference(Q)


def make_decision_lr(**kwargs):

    feats = get_local_feats(vertex=kwargs["vertex"],
                            sentence=kwargs["sentence"],
                            depths=kwargs["depths"],
                            current_compression=kwargs["current_compression"])

    # note: adds to feats dict

    feats = get_global_feats(vertex=kwargs["vertex"],
                             sentence=kwargs["sentence"],
                             feats=feats,
                             current_compression=kwargs["current_compression"],
                             frontier=kwargs["frontier"],
                             lc=kwargs["len_current_compression"])

    X = kwargs["vectorizer"].transform(feats)
    y = kwargs["clf"].predict(X)[0]
    return y


def preproc(sentence, dependencies="basicDependencies"):
    '''
    ILP needs enhanced dependencies b/c of tree transform

    vertex addition method really needs basic dependencies b/c makes strong
    assumptions about tree structure
    '''
    sentence["q"] = set(sentence["q"])

    def get_feats_included(ix):
        out = []
        out.append(("has_already", ix2pos[ix]))
        for c in ix2children[ix]:
            out.append(("has_already_d", c))
        for c in ix2parent[ix]:
            out.append(("has_already_d_dep", c))
        return out

    def get_mark(sentence):
        has_mark_or_xcomp = [_["dep"] in ["mark", "xcomp", "auxpass"] for _ in sentence[dependencies]]
        return any(has_mark_or_xcomp)

    ix2children = defaultdict(list)
    ix2parent = defaultdict(list)
    gov_dep_lookup = {}
    dep2gov = {}
    vx2children = defaultdict(list)
    vx2gov = defaultdict()
    childrencount = defaultdict(int)
    ix2tok = {}
    ix2pos = {}
    neighbors = {}
    gov2deps = defaultdict(set)
    indexes = set()

    for i in sentence[dependencies]:
        dep2gov[i['dependent']] = i['governor']
        gov2deps[i["governor"]].add(i["dependent"])
        ix2children[i["governor"]].append(i["dep"])
        ix2parent[i["dependent"]].append(i["dep"])
        gov_dep_lookup[i["governor"], i["dependent"]] = i
        vx2children[i["governor"]].append(i)
        vx2gov[i["dependent"]] = i
        childrencount[i["governor"]] += 1

    sentence["gov2deps"] = gov2deps

    ix2feats_included = {}
    for tno, t in enumerate(sentence["tokens"]):
        sentence["tokens"][tno]["len"] = len(t["word"])
        ix2tok[t["index"]] = t
        indexes.add(t["index"])
        ix2pos[t["index"]] = t["pos"]
        ix2feats_included[t["index"]] = get_feats_included(t["index"])
        b = gov2deps[t["index"]]
        b.add(dep2gov[t["index"]])
        neighbors[t["index"]] = b
    sentence["neighbors"] = neighbors
    sentence["indexes"] = indexes
    sentence["ix2tok"] = ix2tok
    sentence["gov_dep_lookup"] = gov_dep_lookup
    sentence["ix2pos"] = ix2pos
    sentence["ix2parent"] = ix2parent
    sentence["ix2children"] = ix2children
    sentence["vx2children"] = vx2children
    sentence["vx2gov"] = vx2gov
    sentence["depths"] = get_depths(sentence)
    sentence["childrencount"] = childrencount
    sentence["dep2gov"] = dep2gov
    sentence["feats_included"] = ix2feats_included

    # does not seem to really matter much... sentence["is_root_and_mark_or_xcomp"] = get_mark(sentence)

    sentence["lsentence"] = len(sentence["original"])
    sentence["lqwords"] = sum(_["len"] for _ in sentence["tokens"] if _["index"] in sentence["q"]) + len(sentence["q"]) - 1
    sentence["cr_goal"] = sentence["r"]/sentence["lsentence"]
    sentence["q_as_frac_of_cr"] = sentence["lqwords"]/sentence["r"]

def runtime_path(sentence, frontier_selector, clf, vectorizer, decider=make_decision_lr,  marginal=None):
    '''
    Run additive compression, but use a model not oracle to make an addition decision
    
    The model is provided w/ the decider variable. It is either logistic regression or random based on marginal
    '''

    preproc(sentence)


    current_compression =  sentence["q"]


    frontier = init_frontier(sentence, sentence["q"])

    
    lt = len_current_compression(current_compression, sentence)

    depths = sentence["depths"]

    while frontier != NULLSET and lt < sentence["r"]:

        vertex = frontier_selector(frontier=frontier,
                                   current_compression=current_compression,
                                   sentence=sentence)

        y = decider(vertex=vertex,
                    sentence=sentence,
                    depths=depths,
                    current_compression=current_compression,
                    vectorizer=vectorizer,
                    marginal=marginal,
                    clf=clf,
                    frontier=frontier,
                    len_current_compression=lt)

        if y == 1:
            wouldbe = lt + 1 + sentence["ix2tok"][vertex]["len"]
            if wouldbe <= sentence["r"]:
                current_compression.add(vertex)
                lt = wouldbe

        frontier.remove(vertex)

    return current_compression


def get_local_feats(vertex, sentence, depths, current_compression):
    '''get the features that are local to the vertex to be added'''
    governor = sentence["vx2gov"][vertex]['governor']

    if governor in current_compression:
        #assert vertex not in current_compression /// this does not apply w/ full frontier
        return featurize_child_proposal(sentence,
                                         dependent_vertex=vertex,
                                         governor_vertex=governor,
                                         depths=depths)

    elif len(sentence["gov2deps"][vertex] & current_compression) > 0:
        return featurize_governor_proposal(sentence=sentence,
                                            dependent_vertex=vertex,
                                            depths=depths)

    else:
        return {"type":"DISCONNECTED", "dep": sentence["vx2gov"][vertex]["dep"]}


def featurize_child_proposal(sentence, dependent_vertex, governor_vertex, depths):
    '''return features for a vertex that is a dependent of a vertex in the tree'''
    
    child = sentence["gov_dep_lookup"][governor_vertex, dependent_vertex]

    out = get_features_of_dep(dep=child, sentence=sentence)

    out["type"] = "CHILD"

    return out


def featurize_governor_proposal(sentence, dependent_vertex, depths):
    '''get the features of the proposed governor'''

    out = get_features_of_dep(dep=sentence["vx2gov"][dependent_vertex],
                              sentence=sentence)

    out["type"] = "GOVERNOR"

    return out


def get_connected2(sentence, frontier, current_compression):
    '''get vertexes in frontier that are conntected to current_compression'''

    out = set()
    for ix in current_compression:
        out |= sentence["neighbors"][ix]
    out &= frontier # b/c intersects w/ frontier no need to exclude current compression
    return out


def add_feat(name, val, feats):
    '''also does interactions'''
    feats[name] = val
    feats[name, feats["dep"]] = val # dep + globalfeat
    feats[name, feats["type"]] = val # type (parent/gov/child) + globalfeat
    feats[name, feats["type"], feats["dep"]] = val # type (parent/gov/child) + dep + global feat


def get_global_feats(sentence, feats, vertex, current_compression, frontier, lc):
    '''return global features of the edits'''

    len_tok = sentence["ix2tok"][vertex]["len"]

    # some global features that don't really make sense as interaction feats
    feats["position"] = round(vertex/len(sentence["tokens"]), 1)
    feats["cr_goal"] = sentence["cr_goal"]
    feats["q_as_frac_of_cr"] = sentence["q_as_frac_of_cr"]
    feats["remaining"] = (lc + len_tok)/sentence["r"]

    # If you stop oracle compression once you hit budget this is not really needed 
    # add_feat("over_r", lc + len_tok + 1 > sentence["r"], feats)

    add_feat('middle', vertex > min(current_compression) and vertex < max(current_compression), feats)

    add_feat("r_add", vertex > max(current_compression), feats)

    add_feat("l_add", vertex < min(current_compression), feats)

    ## These seem tempting to cache, but because they are crossed w/ the type feature it is 
    # really annoying to do so. The type of each token (e.g. disconnected or connected)
    # will change as the sentence changes. It would give you like 1ms so perhaps no worth it

    # history based feature. This seems to help w/ 2 points of F1
    for ix in current_compression:
        for f in sentence["feats_included"][ix]:
            add_feat(f, 1, feats)

    return feats


### Other utils. Speed does not matter below here

def has_forest(predicted, sentence):
    ''' is the prediction a forest or a tree?'''
    for p in predicted:
        gov = [_['governor'] for _ in sentence["basicDependencies"] if _["dependent"] == p][0]
        if gov not in predicted | {0}:
            return True
    return False


def get_marginal(fn="training.paths"):
    all_decisions = []
    with open(fn, "r") as inf:
        for ln in inf:
            ln = json.loads(ln)
            for p in ln["paths"]:
                current_compression, vertex, decision, frontier = p
                all_decisions.append(decision)

    return np.mean(all_decisions)


def get_labels_and_features(list_of_paths, only_locals=False):
    '''get the labels and the features from the list of paths'''
    labels = []
    features = []
    for paths in list_of_paths:
        paths = json.loads(paths)
        sentence = paths["sentence"]
        preproc(sentence)
        depths = sentence["depths"]
        for path in paths["paths"]:
            current_compression, vertex, decision, frontier = path
            if vertex != 0:
                feats = get_local_feats(vertex, sentence, depths, set(current_compression))

                # global features
                if not only_locals:
                    lc = len_current_compression(current_compression=current_compression, sentence=sentence)
                    feats = get_global_feats(sentence, feats, vertex, set(current_compression), frontier, lc)

                labels.append(decision)
                features.append(feats)
    return features, labels


def oracle_path(sentence, pi=pick_l2r_connected):
    '''produce all oracle paths, according to policy pi'''
    T = {i for i in sentence["q"]}

    F = init_frontier(sentence, sentence["q"])

    sentence = copy.deepcopy(sentence)

    preproc(sentence)

    #suspected dead decided = []

    path = []
    lt = len_current_compression(T, sentence)

    # does it matter if you add this second AND lt < sentence["r"]?
    # it makes the oracle path slightly more like the runtime path and my intuition is
    # that will be a good thing

    while len(F) > 0 and lt < sentence["r"]:
        v = pi(frontier=F, current_compression=T, sentence=sentence)
        if v in sentence["compression_indexes"]:
            path.append((list(copy.deepcopy(T)), v, 1, list(F)))
            T.add(v)
        else:
            path.append((list(copy.deepcopy(T)), v, 0, list(F)))
        F.remove(v)
        #suspected dead decided.append(v)
        lt = len_current_compression(T, sentence)
    assert T == set(sentence["compression_indexes"])

    return path


def train_clf(training_paths="training.paths",
              validation_paths="validation.paths",
              vectorizer=DictVectorizer(sparse=True, sort=False),
              only_locals=False):
    '''Train a classifier on the oracle path, and check on validation paths'''

    training_paths = [_ for _ in open(training_paths)]
    validation_paths = [_ for _ in open(validation_paths)]

    train_features, train_labels = get_labels_and_features(training_paths, only_locals)

    X_train = vectorizer.fit_transform(train_features)

    y_train = np.asarray(train_labels)

    val_features, val_labels = get_labels_and_features(validation_paths, only_locals)

    X_val = vectorizer.transform(val_features)

    y_val = np.asarray(val_labels)

    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             C=10,
                             multi_class='ovr').fit(X_train, y_train)

    print(clf.score(X_val, y_val))
    print(clf.score(X_train, y_train))

    return clf, vectorizer, clf.predict_proba(X_val)


def get_f1(predicted, jdoc):
    '''get the f1 score for the predicted vertexs vs. the gold'''
    original_ixs = [_["index"] for _ in jdoc["tokens"]]
    y_true = [_ in jdoc['compression_indexes'] for _ in original_ixs]
    y_pred = [_ in predicted for _ in original_ixs]
    return f1_score(y_true=y_true, y_pred=y_pred)


def make_decision_random(**kwargs):
    draw = random.uniform(0, 1)
    return int(draw < kwargs["marginal"]) # ~approx 28% acceptance rate
