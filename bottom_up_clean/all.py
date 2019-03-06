import json
import string
from code.treeops import bfs
import numpy as np
import copy

from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

PUNCT = [_ for _ in string.punctuation]


def get_features_of_dep(dep, sentence, depths):
    '''
    Dep is some dependent in basicDependencies
    '''
    governor_token = get_token_from_sentence(sentence=sentence,
                                             vertex=dep["governor"])

    depentent_token = get_token_from_sentence(sentence=sentence, vertex=dep["dependent"])

    # similar https://arxiv.org/pdf/1510.08418.pdf

    out = defaultdict()
    out["ner_dependent"] = depentent_token["ner"]
    out["pos_dependent"] = depentent_token["pos"]
    out["depth_dependent"] = depths[dep["dependent"]]
    out["position_dependent"] = float(dep["dependent"]/len(sentence["tokens"]))
    out["is_punct_dependent"] = dep["dependentGloss"] in PUNCT
    out["last2_dependent"] = dep["dependentGloss"][-2:]
    out["last2_gov"] = dep["governorGloss"][-2:]
    out["comes_first"] = dep["governor"] < dep["dependent"]
    out["governorGloss"] = dep["governorGloss"]
    out["dependentGloss"] = dep["dependentGloss"]
    out["pos_gov"] = governor_token["pos"]
    out["dep"] = dep["dep"]
    out["governor_in_q"] = dep["governor"] in sentence["q"]

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


def oracle_path_wild_frontier(sentence, pi = pick_l2r_connected):
    T = {i for i in sentence["q"]}
    F = set()
    d, pi_bfs, c = bfs(g=sentence, hop_s=0)

    # init frontier
    for i in sentence["tokens"]:
        if i["index"] not in T:
            F.add(i["index"])
    F.add(0)

    path = []
    while len(F) > 0:
        v = pi(frontier=F, current_compression=T, sentence=sentence)
        if v in sentence["compression_indexes"]:
            for i in get_dependents_and_governors(v, sentence, T):
                F.add(i)
            path.append((copy.deepcopy(T), v, 1))
            T.add(v)
        else:
            path.append((copy.deepcopy(T), v, 0))
        F.remove(v)

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
                             C=.05,
                             multi_class='ovr').fit(X_train, y_train)

    print(clf.score(X_val, y_val))
    print(clf.score(X_train, y_train))
    return clf, vectorizer


def get_depths(sentence):
    '''just a wrapper for bfs that only returns the depths lookup'''
    depths, heads_ignored_var, c_ignored_var = bfs(g=sentence, hop_s=0)
    return depths


def runtime_path_wild_frontier(sentence, frontier_selector, clf, vectorizer):
    current_compression = {i for i in sentence["q"]}
    frontier = set()

    # init frontier
    for i in sentence["tokens"]:
        frontier.add(i["index"])

    depths = get_depths(sentence)

    lt = len_current_compression(current_compression, sentence)

    while len(frontier) > 0 and lt < sentence["r"]:

        vertex = frontier_selector(frontier=frontier,
                                   current_compression=current_compression,
                                   sentence=sentence)

        if vertex != 0: # bug here?
            feats = get_local_feats(vertex=vertex,
                                    sentence=sentence,
                                    depths=depths,
                                    current_compression=current_compression)

            feats = get_global_feats(sentence=sentence,
                                     feats=feats,
                                     vertex=vertex,
                                     current_compression=current_compression)

            X = vectorizer.transform([feats])
            y = clf.predict(X)[0]

            if y == 1:
                current_compression.add(vertex)
                if vertex != 0:
                    for i in get_dependents_and_governors(vertex, sentence, current_compression):
                        if i not in current_compression and i is not None:
                            frontier.add(i)
                else:
                    for i in get_children(sentence, vertex):
                        if i["dependent"] not in current_compression and i is not None:
                            frontier.add(i)
        frontier.remove(vertex)

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


def gov_of_proposed_is_verb_and_current_compression_no_verb(sentence, vertex, current_compression):
    if gov_is_verb(vertex, sentence):
        if not current_compression_has_verb(sentence=sentence,
                                            current_compression=current_compression):
            return True
    return False


def n_verbs_in_s(sentence):
    return sum(1 for i in sentence["tokens"] if i["pos"][0].lower() == "v")


def featurize_disconnected_proposal(sentence, vertex, depths, current_compression, governor):
    # information about the how the proposed disconnected is governed
    feats = featurize_governor_proposal(sentence, dependent_vertex=vertex, depths=depths)

    feats["discon_suffix"] = feats["governorGlossg"][-2:]
    feats["gov_is_root"] = governor == 0
    verby = gov_of_proposed_is_verb_and_current_compression_no_verb(sentence,
                                                                    vertex,
                                                                    current_compression)
    feats["proposed_governed_by_verb"] = verby
    feats["is_next_tok"] = vertex == max(current_compression) + 1

    # if depg is case and is disconnected
    # you need to reason about if the add the pp
    # if there is a lot budget left, you should add the pp. This feat
    # helps reason about this
    if feats["depg"] == "case":
        lt = len_current_compression(current_compression, sentence)
        len_tok = len([_["word"] for _ in sentence["tokens"] if _["index"] == vertex][0])
        feats["remaining_case_discon"] = (lt + len_tok)/sentence["r"]
        grandparent_dep = [_["dep"] for _ in sentence["basicDependencies"] if _["dependent"] == governor]
        if len(grandparent_dep) > 0:
            feats["case_enhanced_deps"] = feats["dependentGlossg"] + ":" + grandparent_dep[0]

        if vertex - 1 in current_compression:
            feats["prev_pos_case"] = [_["pos"] for _ in sentence["tokens"] if _["index"] == vertex - 1][0]

    if (vertex + 1 in current_compression and vertex - 1 in current_compression):
        feats["is_missing"] = 1
    else:
        feats["is_missing"] = 0
    if verby:
        if n_verbs_in_s(sentence) == 1:
            feats["only_verb"] = True
        else:
            feats["only_verb"] = False
        feats["verby_dep"] = feats["depg"]
        if governor != 0:
            gov = get_token_from_sentence(sentence=sentence, vertex=governor)
            feats["gov_discon"] = gov["word"]
            feats["pos_discon"] = gov["pos"]
    feats = {k + "d": v for k, v in feats.items()}
    return feats

def proposed_parent(governor, current_compression):
    '''is the governor in the compression'''
    return governor in current_compression

def proposed_child(current_compression, sentence, vertex):
    dependents = get_children(sentence, vertex)
    return any(d["dependent"] in current_compression for d in dependents)

def get_local_feats(vertex, sentence, depths, current_compression):
    '''get the features that are local to the vertex to be added, and its relationship to the tree #TODO'''
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

    governor["depth"] = depths[governor["governor"]]
    try:
        governor_token = get_token_from_sentence(sentence=sentence, vertex=governor["governor"])
        governor["ner"] = governor_token["ner"]
        governor["pos"] = governor_token["pos"]
    except IndexError: # root
        governor["ner"] = "O"
        governor["pos"] = "O"

    # 0 means governor is root, usually if of the governing verb. Note flip of gov/dep in numerator
    if governor["governor"] == 0:
        governor["position"] = float(governor["dependent"]/len(sentence["tokens"]))
    else:
        governor["position"] = float(governor["governor"]/len(sentence["tokens"]))

    # same interaction feats on gov side
    dependent_token = get_token_from_sentence(sentence=sentence, vertex=governor["dependent"])

    governor["type"] = "GOVERNOR"
    governor["last2"] = governor["dependentGloss"][-2:]
    governor["last2_gov"] = governor["governorGloss"][-2:]
    governor["pos_dep"] = dependent_token["pos"]
    governor["comes_first"] = governor["governor"] < governor["dependent"]
    governor["is_punct"] = governor["governorGloss"] in PUNCT
    governor["parent_label"] = governor["governorGloss"]
    governor["child_label"] = governor["dependentGloss"]
    governor["childrenCount"] = count_children(sentence, governor["governor"])

    features = ["last2", "last2_gov", "pos_dep", "childrenCount", "type", "position",
                "depth", "ner", "pos", "comes_first", "is_punct", "parent_label",
                "child_label"]

    for feat in features:
        governor[feat + governor["dep"]] = governor[feat]

    governor = {k + "g":v for k, v in governor.items() if k not in ["dependent", "governor"]}
    return governor


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


def get_global_feats(sentence, feats, vertex, current_compression):
    '''return global features of the edits'''
    lt = len_current_compression(current_compression, sentence)
    len_tok = len(get_token_from_sentence(sentence, vertex)["word"])
    feats["over_r"] = lt + len_tok > sentence["r"]
    feats["remaining"] = (lt + len_tok)/sentence["r"]

    feats['middle'] = vertex > min(current_compression) and vertex < max(current_compression)

    feats["right_add"] = vertex > max(current_compression)

    feats["left_add"] = vertex < min(current_compression)

    depf = "dep"
    if "depg" in feats:
        depf = "depg"
    elif "depgd" in feats:
        depf = "depgd"
    else:
        depf = "dep"

    feats['middle_dep'] = str(feats['middle']) + feats[depf]
    feats['right_add_dep'] = str(feats['right_add']) + feats[depf]
    feats["left_add_dep"] = str(feats['left_add']) + feats[depf]

    governor = get_governor(vertex, sentence)
    children = get_children(sentence, vertex)

    governor_dep = [_ for _ in sentence["basicDependencies"] if _["governor"] == governor][0]

    feats["global_gov_depGloss"] = governor_dep["dependentGloss"]
    feats["global_gov_govGloss"] = governor_dep["governorGloss"]
    feats["global_gov_govDep"] = governor_dep["dep"]
    feats["global_children_count"] = count_children(sentence, governor)

    assert type(governor) == int
    if proposed_child(current_compression, sentence, vertex) or proposed_parent(governor=governor, current_compression=current_compression):
        feats["disconnected"] = 0
    else:
        feats["disconnected"] = 1

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
