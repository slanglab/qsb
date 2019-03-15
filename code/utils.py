import re
from code.treeops import dfs
from collections import Counter
from code.treeops import get_walk_from_root
from charguana import get_charset


def get_parent(v, jdoc):
    '''get the parent of v in the jdoc'''
    parent = [_ for _ in jdoc["basicDependencies"]
              if int(_["dependent"]) == int(v)]
    if parent == []:
        return None
    else:
        assert len(parent) == 1 or int(v) == 0
        return parent[0]["governor"]


def get_gold_y(jdoc):
    return [o["index"] in jdoc["compression_indexes"] for o in jdoc["tokens"]]


def get_pred_y(predicted_compression, original_indexes):
    assert all(type(o) == int for o in predicted_compression)
    return [o in predicted_compression for o in original_indexes]

