import json
import string
import numpy as np
import copy
import random
from queue import Queue
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

PUNCT = [_ for _ in string.punctuation]


def bfs(g, hop_s):
    '''
    breadth first search
    Args:
        g: a graph
        hop_s: integer starting vertex, our case a root (i.e. 0)
    Returns:
        - color list of found nodes, c
        - list of nodes and predecessor, pi (predecessor == parent if tree)
        - list of depths, d
    '''
    q = Queue()

    deps = "basicDependencies"

    # ancestors
    pi = {i["dependent"]: None for i in g[deps]}
    # dependents
    d = {i["dependent"]: -1 for i in g[deps]}
    # colors
    c = {i["dependent"]: "W" for i in g[deps]}

    c[hop_s] = "G"
    d[hop_s] = 0

    q.put(hop_s)

    while not q.empty():
        u = q.get()
        for v in [i["dependent"] for i in g[deps] if i["governor"] == u]:
            if c[v] == "W":
                c[v] == "G"
                d[v] = d[u] + 1
                pi[v] = u
                q.put(v)
        c[u] = "B"

    return d, pi, c


def get_marginal(fn="training.paths"):
    all_decisions = []
    with open(fn, "r") as inf:
        for ln in inf:
            ln = json.loads(ln)
            for p in ln["paths"]:
                decision = p[-1]
                all_decisions.append(decision)

    return np.mean(all_decisions)


def get_features_of_dep(dep, sentence, depths):
    '''
    Dep is some dependent in basicDependencies
    '''

    depentent_token = get_token_from_sentence(sentence=sentence, vertex=dep["dependent"])

    # similar https://arxiv.org/pdf/1510.08418.pdf

    out = defaultdict()

    ### These are the F and A (2013) features

    # syntactic
    out["dep"] = dep["dep"]
    out["ner_dependent"] = depentent_token["ner"]
    out["pos_dependent"] = depentent_token["pos"]

    # Structural
    out["childrenCount_gov"] = count_children(sentence, dep["governor"])
    out["childrenCount_child"] = count_children(sentence, dep["governor"])
    out["depth_governor"] = depths[dep["dependent"]]

    # Semantic
    def try_to_get(vx, field):
        try:
            governor_token = get_token_from_sentence(sentence=sentence,
                                                     vertex=vx)
            return governor_token[field]
        except IndexError: # root
            return "O"
    out["ner_gov"] = try_to_get(dep["governor"], "ner")
    out["pos_gov"] = try_to_get(dep["governor"], "pos")
    out["is_neg"] = dep["dep"] == "neg"

    # Lexical
    out["dependentGloss"] = dep["dependentGloss"]

    # Here are some other features
    out["depth_dependent"] = depths[dep["dependent"]]
    out["position_dependent"] = float(dep["dependent"]/len(sentence["tokens"]))
    out["is_punct_dependent"] = dep["dependentGloss"] in PUNCT
    out["is_punct_gov"] = dep["governorGloss"] in PUNCT
    out["last2_dependent"] = dep["dependentGloss"][-2:]
    out["last2_gov"] = dep["governorGloss"][-2:]
    out["comes_first"] = dep["governor"] < dep["dependent"]
    out["governor_in_q"] = dep["governor"] in sentence["q"]
    out["gov_is_upper"] = dep["governorGloss"][0].isupper()
    out["dep_is_upper"] = dep["dependentGloss"][0].isupper()
    out["both_upper"] = out["gov_is_upper"] and out["dep_is_upper"] 

    # 0 means governor is root, usually if of the governing verb. Note flip of gov/dep in numerator
    if dep["governor"] == 0:
        out["position_governor"] = float(dep["dependent"]/len(sentence["tokens"]))
    else:
        out["position_governor"] = float(dep["governor"]/len(sentence["tokens"]))

    return dict(out)


def len_current_compression(current_compression, sentence):
    '''get the character length of the current compression'''
    return sum(len(o['word']) for o in sentence["tokens"] if o["index"] in current_compression)


def pick_l2r_connected(frontier, current_compression, sentence):
    connected = get_connected(sentence, frontier, current_compression)

    if len(connected) > 0:
        options = list(connected)
    else:
        unconnected = [o for o in frontier if o not in connected]
        options = list(unconnected)

    options.sort()
    return options[0]


def oracle_path(sentence, pi = pick_l2r_connected):
    '''produce all oracle paths, according to policy pi'''
    T = {i for i in sentence["q"]}

    F = init_frontier(sentence, sentence["q"])

    decided = []

    path = []
    while len(F) > 0:
        v = pi(frontier=F, current_compression=T, sentence=sentence)
        if v in sentence["compression_indexes"]:
            for i in get_dependents_and_governors(v, sentence, T):
                if i not in decided:
                    F.add(i)
            path.append((copy.deepcopy(T), v, 1))
            T.add(v)
        else:
            path.append((copy.deepcopy(T), v, 0))
        F.remove(v)
        decided.append(v)

    assert T == set(sentence["compression_indexes"])

    return path


def train_clf(training_paths="training.paths", validation_paths="validation.paths", vectorizer=DictVectorizer(sparse=True)):
    '''Train a classifier on the oracle path, and check on validation paths'''
    training_paths = [_ for _ in open(training_paths)]
    validation_paths = [_ for _ in open(validation_paths)]

    train_features, train_labels = get_labels_and_features(training_paths)

    X_train = vectorizer.fit_transform(train_features)

    y_train = np.asarray(train_labels)

    val_features, val_labels = get_labels_and_features(validation_paths)

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
    out = {i["index"] for i in sentence["tokens"] if i["index"] not in Q}
    out.add(0)
    return out


def make_decision_lr(**kwargs):
    feats = get_local_feats(vertex=kwargs["vertex"],
                            sentence=kwargs["sentence"],
                            depths=kwargs["depths"],
                            current_compression=kwargs["current_compression"])

    # note: adds to feats dict
    feats = get_global_feats(vertex=kwargs["vertex"],
                             sentence=kwargs["sentence"],
                             feats=feats,
                             current_compression=kwargs["current_compression"])

    X = kwargs["vectorizer"].transform([feats])
    y = kwargs["clf"].predict(X)[0]
    return y


def make_decision_random(**kwargs):
    draw = random.uniform(0, 1)
    return int(draw < kwargs["marginal"]) # ~approx 28% acceptance rate

def runtime_path(sentence, frontier_selector, clf, vectorizer, decider=make_decision_lr,  marginal=None):
    '''
    Run additive compression, but use a model not oracle to make an addition decision
    
    The model is provided w/ the decider variable. It is either logistic regression or random based on marginal
    '''
    current_compression = {i for i in sentence["q"]}
    frontier = init_frontier(sentence, sentence["q"])

    depths = get_depths(sentence)

    decided = []

    lt = len_current_compression(current_compression, sentence)

    while len(frontier) > 0 and lt < sentence["r"]:

        vertex = frontier_selector(frontier=frontier,
                                   current_compression=current_compression,
                                   sentence=sentence)

        if vertex != 0: # bug here?
            y = decider(vertex=vertex,
                                sentence=sentence,
                                depths=depths,
                                current_compression=current_compression,
                                vectorizer=vectorizer,
                                marginal=marginal,
                                clf=clf)

            if y == 1:
                wouldbe = len_current_compression(current_compression | {vertex}, sentence)
                if wouldbe <= sentence["r"]:
                    current_compression.add(vertex)
                    for i in get_dependents_and_governors(vertex, sentence, current_compression):
                        if i not in current_compression and i is not None:
                            if i not in decided:
                                frontier.add(i)
        frontier.remove(vertex)
        decided.append(vertex)

        lt = len_current_compression(current_compression, sentence)

    return current_compression


def current_compression_has_verb(sentence, current_compression):
    '''returns boolean: does the current compression have a verb?'''
    current_pos = {_["pos"][0].lower() for _ in sentence["tokens"] if _["index"] in current_compression}
    return any(i == "v" for i in current_pos)


def gov_is_verb(vertex, sentence):
    '''returns boolean: is the governor a verb?'''
    gov = get_governor(vertex, sentence)
    if gov is not None and gov is not 0:
        pos = [_['pos'][0].lower() == "v" for _ in sentence["tokens"] if _["index"] == gov][0]
        return pos
    else:
        return False


def get_f1(predicted, jdoc):
    '''get the f1 score for the predicted vertexs vs. the gold'''
    original_ixs = [_["index"] for _ in jdoc["tokens"]]
    y_true = [_ in jdoc['compression_indexes'] for _ in original_ixs]
    y_pred = [_ in predicted for _ in original_ixs]
    return f1_score(y_true=y_true, y_pred=y_pred)


def n_verbs_in_s(sentence):
    return sum(1 for i in sentence["tokens"] if i["pos"][0].lower() == "v")


def featurize_disconnected_proposal(sentence, vertex, depths, current_compression, governor):
    # information about the how the proposed disconnected is governed

    feats = {}

    ## Note: see deadcode/improves_discon_perf.py  Cut some feats that increase F1 a bit b/c hard to justify

    feats["gov_is_root"] = governor == 0

    feats["is_next_tok"] = vertex == max(current_compression) + 1

    if (vertex + 1 in current_compression and vertex - 1 in current_compression):
        feats["is_missing"] = 1
    else:
        feats["is_missing"] = 0

    feats["type"] = "DISCONNECTED"

    return feats

def proposed_parent(governor, current_compression):
    '''is the governor in the compression'''
    return governor in current_compression

def proposed_child(current_compression, sentence, vertex):
    dependents = get_children(sentence, vertex)
    return any(d["dependent"] in current_compression for d in dependents)

def get_local_feats(vertex, sentence, depths, current_compression):
    '''get the features that are local to the vertex to be added'''
    governor = get_governor(vertex, sentence)
    if proposed_parent(governor, current_compression):
        #assert vertex not in current_compression /// this does not apply w/ full frontier
        feats = featurize_child_proposal(sentence,
                                         dependent_vertex=vertex,
                                         governor_vertex=governor,
                                         depths=depths)
    elif proposed_child(current_compression, sentence, vertex):
        feats = featurize_governor_proposal(sentence=sentence,
                                            dependent_vertex=vertex,
                                            depths=depths)

    else:
        feats = featurize_disconnected_proposal(sentence=sentence,
                                                vertex=vertex,
                                                depths=depths,
                                                current_compression=current_compression,
                                                governor=governor)
    return feats


def get_labels_and_features(list_of_paths):
    '''get the labels and the features from the list of paths'''
    labels = []
    features = []
    for paths in list_of_paths:
        paths = json.loads(paths)
        sentence = paths["sentence"]
        depths = get_depths(sentence)
        for path in paths["paths"]:
            current_compression, vertex, decision = path
            if vertex != 0:
                feats = get_local_feats(vertex, sentence, depths, current_compression)

                # global features
                feats = get_global_feats(sentence, feats, vertex, current_compression)

                labels.append(decision)
                features.append(feats)
    return features, labels


def get_governor(vertex, sentence):
    '''return the governor of a vertex'''
    for dep in sentence["basicDependencies"]:
        if dep['dependent'] == vertex:
            return dep['governor']
    assert vertex == 0
    return None


def get_token_from_sentence(sentence, vertex):
    '''Get token from a sentence. Assume token is in the sentence'''
    return [_ for _ in sentence["tokens"] if _["index"] == vertex][0]


def featurize_child_proposal(sentence, dependent_vertex, governor_vertex, depths):
    '''return features for a vertex that is a dependent of a vertex in the tree'''
    child = [_ for _ in sentence["basicDependencies"] if _["governor"] == governor_vertex
             and _["dependent"] == dependent_vertex][0]

    out = get_features_of_dep(dep=child, sentence=sentence, depths=depths)

    #out = {k + "child": v for k,v in out.items()}

    out["type"] = "CHILD"

    kys = list(out.keys())

    for feat in kys:
        out[feat + child["dep"]] = out[feat]

    # exclude the literal dependent and governor from the output
    assert "dependent" not in out
    assert "governor" not in out
    return out

def count_children(sentence, vertex):
    '''returns: count of children of vertex in the parse'''
    return sum(1 for i in sentence["basicDependencies"] if i["governor"] == vertex)


def featurize_governor_proposal(sentence, dependent_vertex, depths):
    '''get the features of the proposed governor'''
    governor = [de for de in sentence['basicDependencies'] if de["dependent"] == dependent_vertex][0]

    out = get_features_of_dep(dep=governor, sentence=sentence, depths=depths)

    features = list(out.keys())

    for feat in features:
        out[feat + governor["dep"]] = out[feat]
    out = {k + "g":v for k, v in out.items()}

    out["type"] = "GOVERNOR"

    return out


def in_compression(vertex, current_compression):
    '''returns bool: is vertex in current compression?'''
    return vertex in current_compression


def get_connected(sentence, frontier, current_compression):
    '''get vertexes in frontier that are conntected to current_compression'''
    out = set()
    for dep in sentence["basicDependencies"]:
        dependent = dep["dependent"]
        governor = dep["governor"]
        if in_compression(dependent, current_compression) and not in_compression(governor, current_compression) and dep['governor'] != 0:
            out.add(governor)
        if in_compression(governor, current_compression) and not in_compression(dependent, current_compression):
            out.add(dependent)
    return {i for i in out if i in frontier}

def get_depf(feats):
    try:
        if "depg" in feats:
            depf = "depg"
        elif "depd" in feats:
            depf = "depd"
        else:
            depf = "dep"
    except KeyError:
        depf = "discon"
    return depf

def get_global_feats(sentence, feats, vertex, current_compression):
    '''return global features of the edits'''

    featsg = {}

    lt = len_current_compression(current_compression, sentence)
    len_tok = len(get_token_from_sentence(sentence, vertex)["word"])

    # these two help. it is showing the method is able to reason about what is left in the compression
    featsg["over_r"] = lt + len_tok > sentence["r"]
    featsg["remaining"] = (lt + len_tok)/sentence["r"]

    featsg['middle'] = vertex > min(current_compression) and vertex < max(current_compression)

    featsg["right_add"] = vertex > max(current_compression)

    featsg["left_add"] = vertex < min(current_compression)

    depf = get_depf(feats)

    governor = get_governor(vertex, sentence)

    governor_dep = [_ for _ in sentence["basicDependencies"] if _["governor"] == governor][0]

    featsg["global_gov_depGloss"] = governor_dep["dependentGloss"]
    featsg["global_gov_govGloss"] = governor_dep["governorGloss"]
    featsg["global_gov_govDep"] = governor_dep["dep"]
    featsg["global_children_count"] = count_children(sentence, governor)

    assert isinstance(governor, int)

    ### do interaction features
    for f in featsg:
        feats[f] = featsg[f]
        try:
            feats[f + feats[depf]] = featsg[f] # dep + globalfeat
            feats[f + feats["type"]] = featsg[f] # type (parent/gov/child) + globalfeat
            feats[f + feats["type"] + feats[depf]] = featsg[f] # type (parent/gov/child) + dep + global feat
        except KeyError:
            pass
    return feats

def get_children(sentence, vertex):
    ''''return the dependents of a vertex in basicDependencies'''
    return [dep for dep in sentence['basicDependencies'] if dep["governor"] == vertex]

def get_dependents_and_governors(vertex, sentence, tree):
    '''add a vertexes children to a queue, sort by prob'''
    assert vertex != 0

    out = []
    for child in get_children(sentence, vertex):
        if child["dependent"] not in tree:
            out.append(child["dependent"])
    governor = [dep for dep in sentence['basicDependencies'] if dep["dependent"] == vertex][0]
    if governor["governor"] not in tree:
        out.append(governor["governor"])
    return out

def has_forest(predicted, sentence):
    ''' is the prediction a forest or a tree?'''
    for p in predicted:
        gov = [_['governor'] for _ in sentence["basicDependencies"] if _["dependent"] == p][0]
        if gov not in predicted | {0}:
            return True
    return False
