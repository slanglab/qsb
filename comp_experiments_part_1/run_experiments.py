import json
from code.printers import pretty_print_conl
from singleop.predictors import FigureEightPredictor


def get_parent(v, jdoc):
    '''get the parent of v in the jdoc'''
    parent = [_ for _ in jdoc["basicDependencies"] if int(_["dependent"]) == int(v)]
    if parent == []:
        return None
    else:
        assert len(parent) == 1 or int(v) == 0
        return parent[0]["governor"]


def get_min_compression(v, jdoc):
    '''
    input:
        v(int) vertex
        jdoc(dict) corenlp jdoc blob
    returns:
        list vertexes from root to v
    '''
    path = []
    while get_parent(v, jdoc) is not None:
        path.append(int(v))
        v = get_parent(v, jdoc)
    last_in_path = path[-1]
    top_of_path = [_["dep"] for _ in jdoc["basicDependencies"] if
                   int(_["dependent"]) == int(last_in_path)][0]
    assert top_of_path.lower() == "root"  # path should end in root
    return list(reversed(path))

with open("tests/fixtures/pakistan_example.json", "r") as inf:
    dt = json.load(inf)["sentences"][0]

predictor = FigureEightPredictor(cache="cache/")

pretty_print_conl(dt)
print get_min_compression(8, dt)

validation_set = []


def ner_to_s(tok):
    if tok["ner"] == "PERSON":
        return "P"
    if tok['ner'] == "LOCATION":
        return "L"
    if tok["ner"] == "ORGANIZATION":
        return "O"
    return "X"


with open("preproc/validation.jsonl", "r") as inf:
    for vno, _ in enumerate(inf):
        validation_set.append(json.loads(_))
        if vno > 1000:
            break

def get_ner_string(jdoc):
    nerstr = "".join([ner_to_s(_) for _ in jdoc["tokens"]]) 
    return nerstr


def get_ner_spans(jdoc):

    a = get_ner_string(jdoc)
    
    out = []

    for i in re.finditer("P+", a):
        out.append(i.span())

    for i in re.finditer("O+", a):
        out.append(i.span())

    for i in re.finditer("L+", a):
        out.append(i.span())
    
    return out
    

import re
for v in validation_set:
    print "***"
    for s,e in get_ner_spans(v):
        print " ".join([i["word"] for i in v["tokens"][s:e]])

for tok in dt["tokens"]:
    v = int(tok["index"])
    dep = [_["dep"] for _ in dt["basicDependencies"] if int(_["dependent"]) == int(v)][0]
    #jdoc, op, vertex, dep, worker_id=0
    if dep.lower() != "root":
        print v
        print predictor.predict_proba(jdoc=dt,
                                      op="prune",
                                      vertex=v,
                                      dep=dep,
                                      worker_id=0
                                     )
        
