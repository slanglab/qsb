'''p3'''
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

from tqdm import tqdm
from code.treeops import get_walk_from_root
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.metrics import f1_score
#from code.printers import pretty_print_conl
from sklearn.metrics import f1_score
from code.treeops import bfs
from code.treeops import dfs
from sklearn.feature_extraction import DictVectorizer
from charguana import get_charset


'''put in init b/c want to run this locally'''
from models import *
from nn.models.bottom_up_simple import *
from nn.dataset_readers.bottom_up_reader import *
from nn.predictors.bottom_up_predictor import *
from nn.models import *
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor


def get_UD2symbols():
    '''
    elmo does character-based representation, 
    so represent the OOV symbols w/ non english chars
    '''
    katakana = list(get_charset('katakana'))
    out = {}
    with open("preproc/ud.txt", "r") as inf:
        for lno, ln in enumerate(inf):
            ln = ln.replace("\n", "")
            out[ln] = katakana[lno]
    return out

def pick_at_random(l):
    return random.sample(l, 1)[0]
     
def get_dependents_and_governors(vx, sentence, tree):
    '''add a vertexes children to a queue, sort by prob'''
    children = [d for d in sentence['basicDependencies'] if d["governor"] == vx]
    governor = [d for d in sentence['basicDependencies'] if d["dependent"] == vx][0]
    out = []
    for c in children:
        if c["dependent"] not in tree:
            out.append(c["dependent"])
    if governor["governor"] not in tree:
        out.append(governor["governor"])
    return out

def get_parent(v, jdoc):
    for _ in jdoc["basicDependencies"]:
        if _["dependent"] == v:
            return _["dep"]
    return "ROOT"


def get_encoded_tokens(dep, v, original_s, t, dep2symbol):
            
    IN = "_π"
    OUT = "_ε"
    BRACKETL = "δL"
    BRACKETR = "δR"
    TARGET = "_τ"

    toks = [_ for _ in original_s["tokens"]]
    toks.append({"word": BRACKETR + dep2symbol[dep], "index": v + .5})
    toks.append({"word": BRACKETL + dep2symbol[dep], "index": v - .5})
    toks.sort(key=lambda x:x["index"])
    out = []
    assert v not in t
    for tok in toks:
        if tok["index"] in t:
            out.append(tok["word"] + IN)
        elif tok["index"] == v:
            out.append(tok["word"] + TARGET)
        else:
            out.append(tok["word"] + OUT)
    return out


def get_instance(original_s, v, y, t, dep2symbol):
    '''
    unknown oracle label is for test time
    '''


    orig_ix = [i["index"] for i in original_s["tokens"]]
    dep = get_parent(v, original_s)
    return {"label": y, 
            "q": original_s['q'],
            "r": original_s["r"],
            "dep": dep,
            "tokens": get_encoded_tokens(dep, v, original_s, t, dep2symbol),
            "original_ix": orig_ix,
            "basicDependencies": original_s["basicDependencies"],
            "compression_indexes": original_s["compression_indexes"]}
    return encoding


def oracle_path(sentence, pi = pick_at_random):
    T = {i for i in sentence["q"]}
    F = set()
    
    # init frontier
    for v in T:
        for i in get_dependents_and_governors(v, sentence, T):
            if i not in T:
                F.add(i)
    path = []
    while len(F) > 0:
        v = pi(F)
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


def get_path_to_root(v, jdoc, root_or_pseudo_root=0):
    def get_parent(v):
        for _ in jdoc["basicDependencies"]:
            if _["dependent"] == v:
                return _["governor"]
        return _["governor"]
    out = [v]
    parent = get_parent(v)
    while parent != root_or_pseudo_root: # if not 0, is a governing verb
        v = parent
        out.append(parent)
        parent = get_parent(v)
    if root_or_pseudo_root != 0:
        out.append(root_or_pseudo_root)
    return out


def min_tree_to_root(jdoc, root_or_pseudo_root=0):
    # if pseudo root is not 0, then root is a governing verb
    return {i for q in jdoc["q"] for i in get_path_to_root(q, jdoc, root_or_pseudo_root)}


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
    pseudo_root = heuristic_extract(jdoc=jdoc)
    tree = min_tree_to_root(jdoc=jdoc, root_or_pseudo_root=pseudo_root)
    last_known_good = tree
    while len_tree(tree=tree, jdoc=jdoc) < jdoc["r"]:
        try:
            append_at_random(tree, jdoc)
            if len_tree(tree=tree, jdoc=jdoc) < jdoc["r"]:
                last_known_good = tree
        except ValueError: # it is possible to back into a corner where there are no V left to add.
                           # in these cases, you cant make compression longer and should just stop
            return last_known_good
    return last_known_good


def print_gold(jdoc):
    gold = jdoc["compression_indexes"]
    print(" ".join([_["word"] for _ in jdoc["tokens"] if _["index"] in gold]))


def print_tree(tree, jdoc):
    tk = [_["word"] for _ in jdoc["tokens"] if _["index"] in tree]
    print(" ".join(tk))


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

    for _ in open(fn, "r"):
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
    if vx != 0:
        governor = [d for d in sentence['basicDependencies'] if d["dependent"] == vx][0]
    else:
        governor = {"dep":"root"}
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
            print("[*] Index error"), # these are mostly parse errors from punct governing parts of the tree.)
            return nops
    return nops


def bottom_up_from_corpus(sentence, **kwargs):
    pseudo_root = heuristic_extract(jdoc=sentence)
    tree = min_tree_to_root(jdoc=sentence, root_or_pseudo_root=pseudo_root)
    q_by_prob = []
    for item in tree:
        add_children_to_q(item, q_by_prob, sentence, tree, dep_probs=kwargs["dep_probs"])

    last_known_good = copy.deepcopy(tree)
    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["dependent"]
            tree.add(new_vx)
            add_children_to_q(new_vx, q_by_prob, sentence, tree, dep_probs=kwargs["dep_probs"])
            remove_from_q(new_vx, q_by_prob, sentence)
            if  len_tree(tree, sentence) < sentence["r"]:
                last_known_good = copy.deepcopy(tree)
        except IndexError:
            print("[*] Index error"), # these are mostly parse errors from punct governing parts of the tree.
            return last_known_good

    return last_known_good


def featurize(sentence):
    out = []
    for t in sentence["tokens"]:
        vx = t["index"]
        if t["index"] in sentence["compression_indexes"]:
            children = [d for d in sentence['basicDependencies'] if d["governor"] == vx if d["dep"] not in ["punct"]]
            governor = [d for d in sentence['basicDependencies'] if d["dependent"] == vx][0]
            for c in children:
                y = c["dependent"] in sentence["compression_indexes"]
                #c = {k + v for k, v in c.items()}
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
    for c in children:
        try:
            c = {k: v for k,v in c.items()}
            c["type"] = "CHILD"
            x = v.transform(c)
            c["prob"] = clf.predict_proba(x)[0][1]
        except KeyError:
            c["prob"] = 0
        if c["dependent"] not in tree:
            q.append(c)
    q.sort(key=lambda x: x["prob"], reverse=True)


def bottom_up_from_clf(sentence, **kwargs):
    pseudo_root = heuristic_extract(jdoc=sentence)
    tree = min_tree_to_root(jdoc=sentence, root_or_pseudo_root=pseudo_root)
    clf, v = kwargs["clf"], kwargs["v"]
    q_by_prob = []
    for item in tree:
        add_children_to_q_lr(item, q_by_prob, sentence, tree, clf, v)

    last_known_good = copy.deepcopy(tree)
    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["dependent"]
            tree.add(new_vx)
            add_children_to_q_lr(new_vx, q_by_prob, sentence, tree, clf, v)
            remove_from_q(new_vx, q_by_prob, sentence)
            if len_tree(tree, sentence) < sentence["r"]:
                last_known_good = copy.deepcopy(tree)
        except IndexError:
            print("[*] Index error"), # these are mostly parse errors from punct governing parts of the tree.
            return last_known_good

    return last_known_good


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


class EasyAllenNLP(object):

    def __init__(self, loc="/tmp/548079730"):


        loc = loc
        arch = load_archive(loc, weights_file=loc + "/best.th")
        predictor_name = "bottom_up_predictor"

        self.predictor = Predictor.from_archive(arch, predictor_name)

    def predict_proba(self, paper_json):

        paper_json = {"tokens": [{'word': "hi"}, {"word": "bye"}]}

        sentence = " ".join([_["word"] for _ in paper_json["tokens"]])

        instance = self.predictor._dataset_reader.text_to_instance(sentence)

        pred = self.predictor.predict_instance(instance)

        return pred["class_probabilities"][1]


def add_children_to_q_nn(vx, q, sentence, tree, nn, dep2symbol):
    '''add a vertexes children to a queue, sort by prob'''
    children = [d for d in sentence['basicDependencies'] if d["governor"] == vx if d["dependent"] not in tree]    
    for c in children:
        try: 
            c["type"] = "CHILD"
            paper_json = get_instance(original_s=sentence, v=c["dependent"], y=None, t=tree, dep2symbol=dep2symbol)
            c["prob"] = nn.predict_proba(paper_json)
        except KeyError:
            c["prob"] = 0
        if c["dependent"] not in tree:
            q.append(c)
    q.sort(key=lambda x: x["prob"], reverse=True)


def bottom_up_from_nn(sentence, **kwargs):
    pseudo_root = heuristic_extract(jdoc=sentence)
    tree = min_tree_to_root(jdoc=sentence, root_or_pseudo_root=pseudo_root)
    nn = kwargs["nn"]
    dep2symbol = get_UD2symbols()
    q_by_prob = []
    for item in tree:
        add_children_to_q_nn(item, q_by_prob, sentence, tree, nn, dep2symbol)

    last_known_good = copy.deepcopy(tree)
    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["dependent"]
            tree.add(new_vx)
            add_children_to_q_nn(new_vx, q_by_prob, sentence, tree, nn, dep2symbol)
            remove_from_q(new_vx, q_by_prob, sentence)
            if len_tree(tree, sentence) < sentence["r"]:
                last_known_good = copy.deepcopy(tree)
        except IndexError:
            print("[*] Index error"), # these are mostly parse errors from punct governing parts of the tree.
            return last_known_good

    return last_known_good
