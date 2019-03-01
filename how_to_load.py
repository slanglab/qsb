# coding: utf-8
import pdb

from all import EasyAllenNLP


def add_children_to_q_nn(vx, q, sentence, tree, nn):
    '''add a vertexes children to a queue, sort by prob'''
    children = [d for d in sentence['basicDependencies'] if d["governor"] == vx if d["dep"] not in ["punct"]]    
    for c in children:
        try:
            c = {"c" + k: v for k,v in c.items()}
            c["type"] = "CHILD"
            // transform tokeens
            //  paper_json = {"tokens": [{'word': "hi"}, {"word": "bye"}]}
            c["prob"] = clf.predict_proba(paper_json)
        except KeyError:
            c["prob"] = 0
        if c["cdependent"] not in tree:
            q.append(c)
    q.sort(key=lambda x: x["prob"], reverse=True)


def bottom_up_from_nn(sentence, **kwargs):
    pseudo_root = heuristic_extract(jdoc=sentence)
    tree = min_tree_to_root(jdoc=sentence, root_or_pseudo_root=pseudo_root)
    nn = kwargs["EasyAllenNLP"]
    q_by_prob = []
    for item in tree:
        add_children_to_q_nn(item, q_by_prob, sentence, tree, nn)

    last_known_good = copy.deepcopy(tree)
    while len_tree(tree, sentence) < sentence["r"]:
        try:
            new_vx = q_by_prob[0]["cdependent"]
            tree.add(new_vx)
            add_children_to_q_nn(new_vx, q_by_prob, sentence, tree, nn)
            remove_from_q_nn(new_vx, q_by_prob, sentence)
            if len_tree(tree, sentence) < sentence["r"]:
                last_known_good = copy.deepcopy(tree)
        except IndexError:
            print("[*] Index error"), # these are mostly parse errors from punct governing parts of the tree.
            return last_known_good

    return last_known_good


m = EasyAllenNLP()
print(m.predict_proba()) 