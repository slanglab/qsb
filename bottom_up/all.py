from __future__ import division
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.metrics import f1_score
from code.printers import pretty_print_conl
from sklearn.metrics import f1_score
from code.treeops import bfs
from sklearn.feature_extraction import DictVectorizer


def heuristic_extract(jdoc):
    '''
    return the lowest vertex in the tree that contains the query terms
    '''
    from_root = [_['dependent'] for _ in jdoc["basicDependencies"] if _['governor'] == 0][0]
    best = from_root
    def tok_is_verb(vertex):
        gov = [o["pos"][0] for o in jdoc["tokens"] if o["index"] == v][0]
        return gov[0].lower() == "v"
    for v in get_walk_from_root(jdoc):  # bfs
        children = dfs(g=jdoc, hop_s=v, D=[])
        # the verb heuristic is b/c the min governing tree is often just Q itself
        if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
            best = v
    return best


def get_path_to_root(v, jdoc):
    def get_parent(v):
        for _ in jdoc["basicDependencies"]:
            if _["dependent"] == v:
                return _["governor"]
        return _["governor"]
    out = [v]
    parent = get_parent(v)
    while parent != 0:
        v = parent
        out.append(parent)
        parent = get_parent(v)
    return out


def min_tree_to_root(jdoc):
    return {i for q in jdoc["q"] for i in get_path_to_root(q, jdoc)}


def len_tree(tree, jdoc):
    return sum(len(o['word']) for o in jdoc["tokens"] if o["index"] in tree)


def get_options(tree, jdoc):
    optionsd = {o["dependent"] for o in jdoc["basicDependencies"] if o["governor"] in tree and o["dependent"] not in tree}
    optionsg = {o["governor"] for o in jdoc["basicDependencies"] if o["dependent"] in tree and o["governor"] not in tree}
    return optionsd | optionsg


def append_at_random(tree, jdoc):
    s = get_options(tree, jdoc)
    added = random.sample(s, 1)[0]
    assert added not in tree
    tree.add(added)


def bottom_up_compression_random(jdoc, **kwargs):
    tree = min_tree_to_root(jdoc=jdoc)
    while len_tree(tree=tree, jdoc=jdoc) < jdoc["r"]:
        try:
            append_at_random(tree, jdoc)
        except ValueError: # it is possible to back into a corner where there are no V left to add.
                           # in these cases, you cant make compression longer and should just stop
            return tree
    return tree


def print_gold(jdoc):
    gold = jdoc["compression_indexes"]
    print " ".join([_["word"] for _ in jdoc["tokens"] if _["index"] in gold])


def print_tree(tree, jdoc):
    tk = [_["word"] for _ in jdoc["tokens"] if _["index"] in tree]
    print " ".join(tk)


def get_f1(predicted, jdoc):
    original_ixs = [_["index"] for _ in jdoc["tokens"]]
    y_true = [_ in jdoc['compression_indexes'] for _ in original_ixs]
    y_pred = [_ in predicted for _ in original_ixs]
    return f1_score(y_true=y_true, y_pred=y_pred)


def f1_experiment(sentence_set, f, **kwargs):
    tot = 0
    for sentence in sentence_set:
        predicted = f(sentence, **kwargs)
        tot += get_f1(predicted, sentence)
    return tot/len(sentence_set)


def train_from_corpus(fn):

    dep_counter = defaultdict(list)

    from tqdm import tqdm_notebook as tqdm

    for _ in tqdm(open(fn, "r")):
        _ = json.loads(_)
        toks = [i for i in _["tokens"] if i["index"] in _["compression_indexes"] + [0]]
        for t in toks:
            gov = [d["dep"] for d in _["basicDependencies"] if d["dependent"] == t["index"]]
            assert len(gov) == 1
            gov = gov[0]
            dep = [d["dep"] for d in _["basicDependencies"] if d["governor"] == t["index"]]
            for d in dep:
                dep_counter[gov].append(d)

    from collections import Counter

    dep_probs = defaultdict()
    for d in dep_counter:
        c = Counter(dep_counter[d])
        c = {k: v/sum(c.values()) for k, v in c.items()}
        dep_probs[d] = c
    return dep_probs


def add_children_to_q(vx, q, sentence, tree, dep_probs):
    '''add a vertexes children to a queue, sort by prob'''
    children = [d for d in sentence['basicDependencies'] if d["governor"] == vx if d["dep"] not in ["punct"]]
    governor = [d for d in sentence['basicDependencies'] if d["dependent"] == vx][0]
    for c in children:
        try:
            c["prob"] = dep_probs[governor["dep"]][c["dep"]]
        except KeyError:
            c["prob"] = 0
        if c["dependent"] not in tree:
            q.append(c)
    q.sort(key=lambda x: x["prob"], reverse=True)


def remove_from_q(vx, Q, sentence):
    '''add a vertexes children to a queue, sort by prob'''
    for ino, i in enumerate(Q):
        if i["dependent"] == vx:
            del Q[ino]
            break


def bottom_up_from_corpus_nops(sentence, **kwargs):
    tree = min_tree_to_root(jdoc=sentence)
    q_by_prob = []
    for item in tree:
        add_children_to_q(item, q_by_prob, sentence, tree, dep_probs=kwargs["dep_probs"])

    nops = 0
    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["dependent"]
            tree.add(new_vx)
            add_children_to_q(new_vx, q_by_prob, sentence, tree, dep_probs=kwargs["dep_probs"])
            remove_from_q(new_vx, q_by_prob, sentence)
            nops += 1
        except IndexError:
            print "[*] Index error", # these are mostly parse errors from punct governing parts of the tree.
            return nops
    return nops


def bottom_up_from_corpus(sentence, **kwargs):
    tree = min_tree_to_root(jdoc=sentence)
    q_by_prob = []
    for item in tree:
        add_children_to_q(item, q_by_prob, sentence, tree, dep_probs=kwargs["dep_probs"])

    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["dependent"]
            tree.add(new_vx)
            add_children_to_q(new_vx, q_by_prob, sentence, tree, dep_probs=kwargs["dep_probs"])
            remove_from_q(new_vx, q_by_prob, sentence)
        except IndexError:
            print "[*] Index error" # these are mostly parse errors from punct governing parts of the tree.
            return tree

    return tree


def featurize(sentence):
    out = []
    for t in sentence["tokens"]:
        vx = t["index"]
        if t["index"] in sentence["compression_indexes"]:
            children = [d for d in sentence['basicDependencies'] if d["governor"] == vx if d["dep"] not in ["punct"]]
            governor = [d for d in sentence['basicDependencies'] if d["dependent"] == vx][0]
            for c in children:
                y = c["dependent"] in sentence["compression_indexes"]
                c = {k + "c": v for k, v in c.items()}
                c["type"] = "CHILD"
                feats = c
                out.append({"feats": feats, "y": y})
            assert governor["dependent"] in sentence["compression_indexes"]
            y = governor["governor"] in sentence["compression_indexes"]
            governor = {k + "g":v for k,v in governor.items()}
            governor["type"] = "GOVERNOR"
            feats = governor
            #out.append({"feats": feats, "y": y}) no gov feats at the moment
    return out


def get_features_and_labels(fn, cutoff=10000000):

    out = []

    for sno, sentence in enumerate(open(fn)):
        if sno == cutoff:
            break
        sentence = json.loads(sentence)
        out = out + featurize(sentence)

    return out


def add_children_to_q_lr(vx, q, sentence, tree, clf, v):
    '''add a vertexes children to a queue, sort by prob'''
    children = [d for d in sentence['basicDependencies'] if d["governor"] == vx if d["dep"] not in ["punct"]]
    governor = [d for d in sentence['basicDependencies'] if d["dependent"] == vx][0]
    for c in children:
        try:
            c = {"c" + k: v for k,v in c.items()}
            c["type"] = "CHILD"
            x = v.transform(c)
            c["prob"] = clf.predict_proba(x)[0][1]
        except KeyError:
            c["prob"] = 0
        if c["cdependent"] not in tree:
            q.append(c)
    q.sort(key=lambda x: x["prob"], reverse=True)


def remove_from_q_lr(vx, Q, sentence):
    '''add a vertexes children to a queue, sort by prob'''
    for ino, i in enumerate(Q):
        if i["cdependent"] == vx:
            del Q[ino]
            break


def bottom_up_from_clf(sentence, **kwargs):
    clf, v = kwargs["clf"], kwargs["v"]
    tree = min_tree_to_root(jdoc=sentence)
    q_by_prob = []
    for item in tree:
        add_children_to_q_lr(item, q_by_prob, sentence, tree, clf, v)

    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["cdependent"]
            tree.add(new_vx)
            add_children_to_q_lr(new_vx, q_by_prob, sentence, tree, clf, v)
            remove_from_q_lr(new_vx, q_by_prob, sentence)
        except IndexError:
            print "[*] Index error", # these are mostly parse errors from punct governing parts of the tree.
            return tree

    return tree


def plot_slens(slens):
    x, y = [], []

    for s in slens:
        x.append(s)
        y.append(np.mean(slens[s]))

    plt.scatter(x, y)
    plt.title("Empirical complexity by sentence length")
    plt.ylabel("Total vertexes eval for pruning")
    plt.xlabel("Sentence length")
    plt.show()


def get_slens(dev, f, **kwargs):
    slens = defaultdict(list)
    for d in dev:
        slen = len(d["tokens"])
        if slen < 30:
            nops = f(d, **kwargs)  # get nops
            slens[slen].append(nops)
    return slens
