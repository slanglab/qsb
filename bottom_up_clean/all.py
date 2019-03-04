import json
import string
from code.treeops import bfs
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction import DictVectorizer

PUNCT = [_ for _ in string.punctuation]

def len_current_compression(current_compression, sentence):
    '''get the character length of the current compression'''
    return sum(len(o['word']) for o in sentence["tokens"] if o["index"] in current_compression)

def train_clf(training_paths = "training.paths", validation_paths="validation.paths"):
    '''Train a classifier on the oracle path, and check on validation paths'''
    training_paths = [_ for _ in open(training_paths)]
    validation_paths = [_ for _ in open(validation_paths)]

    vectorizer = DictVectorizer(sparse=True)

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
    depths, heads_ignored_var, c_ignored_var = bfs(g=sentence, hop_s=0)
    return depths


def runtime_path_wild_frontier(sentence, frontier_selector, clf, vectorizer):
    current_compression = {i for i in sentence["q"]}
    frontier = set()

    # init frontier
    for i in sentence["tokens"]:
        frontier.add(i["index"])

    depths = get_depths(sentence)

    while len(frontier) > 0:
        vertex = frontier_selector(frontier=frontier,
                                   current_compression=current_compression,
                                   sentence=sentence)

        if vertex != 0: # bug here?
            feats = get_local_feats(vertex=vertex, sentence=sentence, depths=depths, current_compression=current_compression)
            feats = get_global_feats(sentence=sentence, feats=feats, vertex=vertex, current_compression=current_compression)

            X = vectorizer.transform([feats])
            y = clf.predict(X)[0]

            if y == 1:
                current_compression.add(vertex)
                if vertex != 0:
                    for i in get_dependents_and_governors(vertex, sentence, current_compression):
                        if i not in current_compression and i is not None:
                            frontier.add(i)
                else:
                    for i in get_dependents(sentence, vertex):
                        if i["dependent"] not in current_compression and i is not None:
                            frontier.add(i)
        frontier.remove(vertex)

    return current_compression


def pick_l2r_connected(frontier, current_compression, sentence):
    connected = get_connected(sentence, frontier, current_compression)
    unconnected = [o for o in frontier if o not in connected]

    if len(connected) > 0:
        options = list(connected)
    else:
        options = list(unconnected)

    options.sort()
    return options[0]

def current_compression_has_verb(sentence, current_compression):
    current_pos = {_["pos"][0].lower() for _ in sentence["tokens"] if _["index"] in current_compression}
    return any(i == "v" for i in current_pos)

def gov_is_verb(vertex, sentence):
    gov = get_governor(vertex, sentence)
    if gov is not None and gov is not 0:
        pos = [_['pos'][0].lower() == "v" for _ in sentence["tokens"] if _["index"] == gov][0]
        return pos
    else:
        return False

def get_f1(predicted, jdoc):
    original_ixs = [_["index"] for _ in jdoc["tokens"]]
    y_true = [_ in jdoc['compression_indexes'] for _ in original_ixs]
    y_pred = [_ in predicted for _ in original_ixs]
    return f1_score(y_true=y_true, y_pred=y_pred)

def gov_of_proposed_is_verb_and_current_compression_no_verb(sentence, vertex, current_compression):
    if gov_is_verb(vertex, sentence) and not current_compression_has_verb(sentence=sentence, current_compression=current_compression):
        return True
    else:
        return False


def n_verbs_in_s(sentence):
    return sum(1 for i in sentence["tokens"] if i["pos"][0].lower() == "v")

def get_local_feats(vertex, sentence, depths, current_compression):
    governor = get_governor(vertex, sentence)
    dependents = get_dependents(sentence, vertex)
    if governor in current_compression:
        #assert vertex not in current_compression /// this does not apply w/ full frontier
        feats = featurize_child_proposal(sentence, dependent_vertex=vertex, governor_vertex=governor, depths=depths)
        feats["disconnected"] = 0
    elif any(d["dependent"] in current_compression for d in dependents):
        feats = featurize_parent_proposal(sentence, dependent_vertex=vertex, depths=depths)
        feats["disconnected"] = 0

    else:
        # information about the how the proposed disconnected is governed
        feats = featurize_parent_proposal(sentence, dependent_vertex=vertex, depths=depths)
        feats["discon_suffix"] = feats["governorGlossg"][-2:]
        feats = {k + "d": v for k, v in feats.items()}
        feats["disconnected"] = 1
        feats["gov_is_root"] = governor == 0
        verby = gov_of_proposed_is_verb_and_current_compression_no_verb(sentence, vertex, current_compression)
        feats["proposed_governed_by_verb"] = verby
        feats["is_next_tok"] = vertex == max(current_compression) + 1

        # if depg is case and is disconnected
        # you need to reason about if the add the pp
        # if there is a lot budget left, you should add the pp. This feat
        # helps reason about this
        if feats["depgd"] == "case":
            lt = len_current_compression(current_compression, sentence)
            len_tok = len([_["word"] for _ in sentence["tokens"] if _["index"] == vertex][0])
            feats["remaining_case_discon"] = (lt + len_tok)/sentence["r"]
            grandparent_dep = [_["dep"] for _ in sentence["basicDependencies"] if _["dependent"] == governor]
            if len(grandparent_dep) > 0:
                feats["case_enhanced_deps"] = feats["dependentGlossgd"] + ":" + grandparent_dep[0]

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
            feats["verby_dep"] = feats["depgd"]
            if governor != 0:
                gov = [_ for _ in sentence["tokens"] if _["index"] == governor][0]
                feats["gov_discon"] = gov["word"]
                feats["pos_discon"] = gov["pos"]

    return feats


def get_labels_and_features(list_of_paths):
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
    for dep in sentence["basicDependencies"]:
        if dep['dependent'] == vertex:
            return dep['governor']
    assert vertex == 0
    return None


def get_token_from_sentence(sentence, vertex):
    '''Get token from a sentence. Assume token is in the sentence'''
    return [_ for _ in sentence["tokens"] if _["index"] == vertex][0]

def featurize_child_proposal(sentence, dependent_vertex, governor_vertex, depths):
    child = [_ for _ in sentence["basicDependencies"] if _["governor"] == governor_vertex and _["dependent"] == dependent_vertex][0]

    child["type"] = "CHILD"

    # crossing this w/ dep seems to lower F1
    child["position"] = float(child["dependent"]/len(sentence["tokens"]))

    if child["governor"] in sentence["q"]:
        child["compund_off_q"] = True

    child["is_punct"] = child["dependentGloss"] in PUNCT
    child["last2_"] = child["dependentGloss"][-2:]
    child["last2_gov"] = child["governorGloss"][-2:]
    child["pos_gov_"] = [_["pos"] for _ in sentence["tokens"] if _["index"] == child["governor"]][0]
    child["comes_first"] = child["governor"] < child["dependent"]

    child["is_punct" + child["dep"]] = child["dependentGloss"] in PUNCT
    child["last2_" + child["dep"]] = child["dependentGloss"][-2:]
    child["last2_gov" + child["dep"]] = child["governorGloss"][-2:]
    child["pos_gov_" + child["dep"]] = [_["pos"] for _ in sentence["tokens"] if _["index"] == child["governor"]][0]
    child["comes_first" + child["dep"]] = child["governor"] < child["dependent"]

    # similar https://arxiv.org/pdf/1510.08418.pdf
    child["parent_label"] = child["dep"] + child["governorGloss"]
    child["child_label"] = child["dep"] + child["dependentGloss"]
    depentent_token = get_token_from_sentence(sentence=sentence, vertex=child["dependent"])
    child["ner"] = depentent_token["ner"]
    child["pos"] = depentent_token["pos"]
    child["depth"] = depths[child["dependent"]]
    child = {k:v for k, v in child.items() if k not in ["dependent", "governor"]}
    feats = child
    return feats

def count_children(sentence, vertex):
    '''returns: count of children of vertex in the parse'''
    return sum(1 for i in sentence["basicDependencies"] if i["governor"] == vertex)


def featurize_parent_proposal(sentence, dependent_vertex, depths):
    governor = [de for de in sentence['basicDependencies'] if de["dependent"] == dependent_vertex][0]

    governor["depth"] = depths[governor["governor"]]
    try:
        governor["ner"] = [_["ner"] for _ in sentence["tokens"] if _["index"] == governor["governor"]][0]
    except IndexError: # root
        governor["ner"] = "O"

    try:
        governor["pos"] = [_["pos"] for _ in sentence["tokens"] if _["index"] == governor["governor"]][0]
    except IndexError: # root
        governor["pos"] = "O"

    if governor["governor"] == 0: # dep of root, usually governing verb. note flip gov/dep in numerator
        governor["position"] = float(governor["dependent"]/len(sentence["tokens"]))
    else:
        governor["position"] = float(governor["governor"]/len(sentence["tokens"]))

    governor["is_punct"] = governor["governorGloss"] in PUNCT

    governor["parent_label"] = governor["governorGloss"]
    governor["child_label"] = governor["dependentGloss"]
    governor["type"] = "GOVERNOR"

    # same interaction feats on gov side
    governor["last2"] = governor["dependentGloss"][-2:]
    governor["last2_gov"] = governor["governorGloss"][-2:]
    governor["pos_dep"] = [_["pos"] for _ in sentence["tokens"] if _["index"] == governor["dependent"]][0]
    governor["comes_first"] = governor["governor"] < governor["dependent"]

    governor["last2_" + governor["dep"]] = governor["dependentGloss"][-2:]
    governor["last2_gov" + governor["dep"]] = governor["governorGloss"][-2:]
    governor["pos_dep_" + governor["dep"]] = [_["pos"] for _ in sentence["tokens"] if _["index"] == governor["dependent"]][0]
    governor["comes_first" + governor["dep"]] = governor["governor"] < governor["dependent"]
    governor["is_punct"  + governor["dep"]] = governor["governorGloss"] in PUNCT

    governor["parent_label_interaction"] = governor["dep"] + governor["governorGloss"]
    governor["child_label_interaction"] = governor["dep"] + governor["dependentGloss"]
    governor["childrenCount"] = count_children(sentence, governor["governor"])
    governor = {k + "g":v for k, v in governor.items() if k not in ["dependent", "governor"]}
    return governor


def in_compression(vertex, current_compression):
    return vertex in current_compression


def get_connected(sentence, frontier, current_compression):
    '''get vx in F that are conntected to T'''
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
    lt = len_current_compression(current_compression, sentence)
    len_tok = len([_["word"] for _ in sentence["tokens"] if _["index"] == vertex][0])
    feats["over_r"] = lt + len_tok > sentence["r"]
    feats["remaining"] = (lt + len_tok)/sentence["r"]

    feats['middle'] = vertex > min(current_compression) and vertex < max(current_compression)

    feats["right_add"] = vertex > max(current_compression)

    feats["left_add"] = vertex < min(current_compression)

    if 'dep' in feats:
        feats['middle_dep'] = str(feats['middle']) + feats["dep"]
        feats['right_add_dep'] = str(feats['right_add']) + feats["dep"]
        feats["left_add_dep"] = str(feats['left_add']) + feats["dep"]

    if 'depg' in feats:
        feats['middle_dep'] = str(feats['middle']) + feats["depg"]
        feats['right_add_dep'] = str(feats['right_add']) + feats["depg"]
        feats["left_add_dep"] = str(feats['left_add']) + feats["depg"]

    return feats

def get_dependents(sentence, vertex):
    return [dep for dep in sentence['basicDependencies'] if dep["governor"] == vertex]

def get_dependents_and_governors(vertex, sentence, tree):
    '''add a vertexes children to a queue, sort by prob'''
    assert vertex != 0
    children = get_dependents(sentence, vertex)
    governor = [dep for dep in sentence['basicDependencies'] if dep["dependent"] == vertex][0]
    out = []
    for child in children:
        if child["dependent"] not in tree:
            out.append(child["dependent"])
    if governor["governor"] not in tree:
        out.append(governor["governor"])
    return out
