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

with open("tests/fixtures/simple_sbar.txt.json", "r") as inf:
    dt = json.load(inf)["sentences"][0]


predictor = FigureEightPredictor(cache="cache/")

pretty_print_conl(dt)
print get_min_compression(8, dt)
