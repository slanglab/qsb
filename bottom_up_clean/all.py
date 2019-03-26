import json
import string
import numpy as np
import copy
import random

from functools import lru_cache

from bottom_up_clean.utils import bfs
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher


def get_siblings(e, jdoc):
    '''
    This gets the other children of h (that are not n). See below

    inputs:
        e(int,int):an edge from head, h, to node n. e[ix] is head 2 tail
        jdoc(dict): a sentence from CoreNLP
    returns
        - other children of h that are not e
    '''
    return [i["dependentGloss"] for i in jdoc["vx2children"][e[0]] if i["dependent"] != e[1]]


def get_features_of_dep(dep, sentence, depths):
    '''
    Dep is some dependent in basicDependencies

    references to h and n come from F & A 2013 table 1
    '''

    depentent_token = sentence["ix2tok"][dep["dependent"]]

    try:
        governor_token = sentence["ix2tok"][dep["governor"]]
    except:
        governor_token = {"lemma": "is_root", "word": "", "len": 0, 
                          "index": 0, "ner": "O", "pos": "root"}

    sibs = get_siblings(e=(dep["governor"], dep["dependent"]), jdoc=sentence)

    out = dict()

    ### These are the F and A (2013) features

    # syntactic
    out["dep"] = dep["dep"]
    for s in sibs:
        out[("ds", s)] = 1
    out["pos_h"] = governor_token["pos"]
    out["pos_n"] = depentent_token["pos"]

    # Structural
    out["childrenCount_h"] = sentence["childrencount"][dep["governor"]]
    out["childrenCount_e"] = sentence["childrencount"][dep["dependent"]]
    out["depth_h"] = depths[dep["dependent"]]
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
    return sum(o["len"] for o in sentence["tokens"] if o["index"] in current_compression) + len(current_compression) - 1


def pick_l2r_connected(frontier, current_compression, sentence):
    connected = get_connected2(sentence, frontier, current_compression)
    if len(connected) > 0:
        options = connected
    else:
        options = list(frontier)

    options.sort()
    return options[0]


def oracle_path(sentence, pi=pick_l2r_connected):
    '''produce all oracle paths, according to policy pi'''
    T = {i for i in sentence["q"]}

    F = init_frontier(sentence, sentence["q"])

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
              feature_config=None):
    '''Train a classifier on the oracle path, and check on validation paths'''

    training_paths = [_ for _ in open(training_paths)]
    validation_paths = [_ for _ in open(validation_paths)]

    train_features, train_labels = get_labels_and_features(training_paths, feature_config)

    X_train = vectorizer.fit_transform(train_features)

    y_train = np.asarray(train_labels)

    val_features, val_labels = get_labels_and_features(validation_paths, feature_config)

    X_val = vectorizer.transform(val_features)

    y_val = np.asarray(val_labels)

    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             C=10,
                             multi_class='ovr').fit(X_train, y_train)

    print(clf.score(X_val, y_val))
    print(clf.score(X_train, y_train))

    return clf, vectorizer, clf.predict_proba(X_val)


def get_depths(sentence):
    '''just a wrapper for bfs that only returns the depths lookup'''
    depths, heads_ignored_var, c_ignored_var = bfs(g=sentence, hop_s=0)
    return depths


def init_frontier(sentence, Q):
    '''initalize the frontier for additive compression'''
    return {i["index"] for i in sentence["tokens"] if i["index"] not in Q}


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
                             frontier=kwargs["frontier"])

    X = kwargs["vectorizer"].transform(feats)
    y = kwargs["clf"].predict(X)[0]
    return y


def make_decision_random(**kwargs):
    draw = random.uniform(0, 1)
    return int(draw < kwargs["marginal"]) # ~approx 28% acceptance rate

def preproc(sentence):

    def get_feats_included(ix):
        out = []
        out.append(("has_already", ix2pos[ix]))
        for c in ix2children[ix]:
            out.append(("has_already_d", c))
        for c in ix2parent[ix]:
            out.append(("has_already_d_dep", c))
        return out

    def get_mark(sentence):
        has_mark_or_xcomp = [_["dep"] in ["mark", "xcomp", "auxpass"] for _ in sentence["basicDependencies"]]
        return any(has_mark_or_xcomp)

    sentence["is_root_and_mark_or_xcomp"] = get_mark(sentence)
    sentence["lsentence"] = len(" ".join([_["word"] for _ in sentence["tokens"]]))
    sentence["lqwords"] = len(" ".join([_["word"] for _ in sentence["tokens"] if _["index"] in sentence["q"]]))
    sentence["cr_goal"] = sentence["r"]/sentence["lsentence"]
    sentence["q_as_frac_of_cr"] = sentence["lqwords"]/sentence["r"]

    ix2children = defaultdict(list)
    ix2parent = defaultdict(list)
    gov_dep_lookup = {}
    dep2gov = {}
    vx2children = defaultdict(list)
    vx2gov = defaultdict()
    childrencount = defaultdict(int)
    ix2tok = {}
    ix2pos = {}

    for i in sentence["basicDependencies"]:
        dep2gov[i['dependent']] = i['governor']
        ix2children[i["governor"]].append(i["dep"])
        ix2parent[i["dependent"]].append(i["dep"])
        gov_dep_lookup["{},{}".format(i["governor"], i["dependent"])] = i
        vx2children[i["governor"]].append(i)
        vx2gov[i["dependent"]] = i
        childrencount[i["governor"]] += 1

    ix2feats_included = {}
    for tno, t in enumerate(sentence["tokens"]):
        sentence["tokens"][tno]["len"] = len(t["word"])
        ix2tok[t["index"]] = t
        ix2pos[t["index"]] = t["pos"]
        ix2feats_included[t["index"]] = get_feats_included(t["index"])

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
    dep2gov[0] = None
    vx2gov[0] = "ROOT"
    sentence["feats_included"] = ix2feats_included


def runtime_path(sentence, frontier_selector, clf, vectorizer, decider=make_decision_lr,  marginal=None):
    '''
    Run additive compression, but use a model not oracle to make an addition decision
    
    The model is provided w/ the decider variable. It is either logistic regression or random based on marginal
    '''
    current_compression = set(sentence["q"])
    frontier = init_frontier(sentence, sentence["q"])

    preproc(sentence)

    lt = len_current_compression(current_compression, sentence)

    depths = sentence["depths"]

    while len(frontier) > 0 and lt < sentence["r"]:

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
                    frontier=frontier)

        if y == 1:
            wouldbe = lt + 1 + sentence["ix2tok"][vertex]["len"]
            if wouldbe <= sentence["r"]:
                current_compression.add(vertex)
                lt = wouldbe

        frontier.remove(vertex)

    return current_compression


def get_f1(predicted, jdoc):
    '''get the f1 score for the predicted vertexs vs. the gold'''
    original_ixs = [_["index"] for _ in jdoc["tokens"]]
    y_true = [_ in jdoc['compression_indexes'] for _ in original_ixs]
    y_pred = [_ in predicted for _ in original_ixs]
    return f1_score(y_true=y_true, y_pred=y_pred)


def get_local_feats(vertex, sentence, depths, current_compression):
    '''get the features that are local to the vertex to be added'''
    governor = sentence["vx2gov"][vertex]['governor']

    if governor in current_compression:
        #assert vertex not in current_compression /// this does not apply w/ full frontier
        feats = featurize_child_proposal(sentence,
                                         dependent_vertex=vertex,
                                         governor_vertex=governor,
                                         depths=depths)

    elif any(d["dependent"] in current_compression for d in sentence["vx2children"][vertex]):
        feats = featurize_governor_proposal(sentence=sentence,
                                            dependent_vertex=vertex,
                                            depths=depths)

    else:
        feats = {"type":"DISCONNECTED", "dep_discon": sentence["vx2gov"][vertex]["dep"]}
    return feats


def get_labels_and_features(list_of_paths, feature_config):
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
                feats = get_local_feats(vertex, sentence, depths, current_compression)

                # global features
                feats = get_global_feats(sentence, feats, vertex, current_compression, frontier)

                labels.append(decision)
                features.append(feats)
    return features, labels


def featurize_child_proposal(sentence, dependent_vertex, governor_vertex, depths):
    '''return features for a vertex that is a dependent of a vertex in the tree'''
    
    child = sentence["gov_dep_lookup"]["{},{}".format(governor_vertex, dependent_vertex)]

    out = get_features_of_dep(dep=child, sentence=sentence, depths=depths)

    out["type"] = "CHILD"

    return out


def featurize_governor_proposal(sentence, dependent_vertex, depths):
    '''get the features of the proposed governor'''

    out = get_features_of_dep(dep=sentence["vx2gov"][dependent_vertex],
                              sentence=sentence,
                              depths=depths)

    out["type"] = "GOVERNOR"

    return out


def get_connected2(sentence, frontier, current_compression):
    '''get vertexes in frontier that are conntected to current_compression'''

    out = []
    for ix in current_compression:
        if sentence["dep2gov"][ix] not in current_compression:
            if sentence["dep2gov"][ix] in frontier:
                out.append(sentence["dep2gov"][ix])
        for i in sentence["vx2children"][ix]:
            if i["dependent"] in frontier:
                out.append(i["dependent"])
    return out


def get_depf(feats):
    if ('dep', 'g') in feats:
        return ('dep', 'g')
    elif "dep_discon" in feats:
        return "dep_discon"
    elif "dep" in feats:
        return "dep"
    else:
        assert "bad" == "thng"


def get_global_feats(sentence, feats, vertex, current_compression, frontier):
    '''return global features of the edits'''

    lt = len_current_compression(current_compression, sentence)
    len_tok = sentence["ix2tok"][vertex]["len"]

    # some global features that don't really make sense as interaction feats
    feats["position"] = round(vertex/len(sentence["tokens"]), 1)
    feats["cr_goal"] = sentence["cr_goal"]
    feats["q_as_frac_of_cr"] = sentence["q_as_frac_of_cr"]
    feats["remaining"] = (lt + len_tok)/sentence["r"]

    depf = get_depf(feats)

    def add_feat(name, val):
        '''also does interactions'''
        feats[name] = val
        feats[name, feats[depf]] = val # dep + globalfeat
        feats[name, feats["type"]] = val # type (parent/gov/child) + globalfeat
        feats[name, feats["type"], feats[depf]] = val # type (parent/gov/child) + dep + global feat

    # these two help. it is showing the method is able to reason about what is left in the compression
    add_feat("over_r", lt + len_tok + 1 > sentence["r"])

    add_feat('middle', vertex > min(current_compression) and vertex < max(current_compression))

    add_feat("r_add", vertex > max(current_compression))

    add_feat("l_add", vertex < min(current_compression))

    governor = sentence["vx2gov"][vertex]['governor']

    add_feat("ggovDep", sentence["vx2children"][governor][0]["dep"])

    # history based feature
    for tok in frontier:
        add_feat(("out_", sentence["ix2pos"][tok]), 1)

    # history based feature
    for ix in current_compression:
        for f in sentence["feats_included"][ix]:
            add_feat(f, 1)

    add_feat("is_root_and_mark_or_xcomp", "is_root_and_mark_or_xcomp")

    return feats


### Other utils

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
                current_compression, vertex, decision, decideds = p
                all_decisions.append(decision)

    return np.mean(all_decisions)
