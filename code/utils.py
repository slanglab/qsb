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

def get_min_compression_Q(Q, jdoc):
    cix = [get_min_compression(v, jdoc) for v in Q]
    return [i for v in cix for i in v]

def ner_to_s(tok):
    if tok["ner"] == "PERSON":
        return "P"
    if tok['ner'] == "LOCATION":
        return "L"
    if tok["ner"] == "ORGANIZATION":
        return "O"
    return "X"


def get_gold_y(jdoc):
    return [o["index"] in jdoc["compression_indexes"] for o in jdoc["tokens"]]


def get_pred_y(predicted_compression, original_indexes):
    assert all(type(o) == int for o in predicted_compression)
    return [o in predicted_compression for o in original_indexes]


def get_ner_string(jdoc):
    '''make a big string showing NER tag at each position'''
    nerstr = "".join([ner_to_s(_) for _ in jdoc["tokens"]])
    return nerstr


def is_prune_only(jdoc):
    one_extract = Counter(jdoc["oracle"].values())["e"] == 1
    extract_v = int([k for k, v in jdoc["oracle"].items()
                    if v == "e"][0])
    gov = [i["governor"] for i in jdoc["basicDependencies"] if
           i["dependent"] == extract_v][0]
    return gov == 0 and one_extract


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


def get_ner_spans_in_compression(v):
    c_ix = v["compression_indexes"]
    out = []
    for s, e in get_ner_spans(v):
        ner_toks = [i["index"] for i in v["tokens"][s:e]]
        if all(i in c_ix for i in ner_toks):
            out.append(ner_toks)
    return out


def get_NER_query(jdoc):
    '''get the query toks for the NER'''
    c_ix = jdoc["compression_indexes"]
    out = []
    for s, e in get_ner_spans(jdoc):
        ner_toks = [i["index"] for i in jdoc["tokens"][s:e]]
        if all(i in c_ix for i in ner_toks):
            for j in ner_toks:
                out.append(int(j))
    return out


def getUD2symbols():
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


ud2symbols = getUD2symbols()


def get_labeled_toks(node, jdoc, op_proposed):
    if len(jdoc["tokens"]) == 0:
        labeled_toks = [{"word": "α", "index": -1000},{"word": "Ω", "index":1000}]
        return labeled_toks

    dep = [_["dep"] for _ in jdoc["basicDependencies"] if _["dependent"] == node][0]
    
    #learning curve seems to be capping at 4M examples. but if you add more, uncomment this line to fill any unseen deps before rerunning.    
    #with open("preproc/ud.txt", "a") as of:
    #    of.write(dep + "\n")
    
    START = "β" + ud2symbols[dep]  +op_proposed   # BETA is BracketStart
    END = "γ" + ud2symbols[dep] + op_proposed  # gamma is BracketEnd
    toks = [i for i in jdoc["tokens"]]
    cut = dfs(g=jdoc, hop_s=node, D=[])
    cut.sort()
    mint = min(cut)
    maxt = max(cut)
    # This assertion is false in cases where you greedily prune in the middle of 
    # trees. it is true if you only only prune a whole branch 
    #assert len(cut) == len(range(mint, maxt + 1))

    # the indexes are added w/ +.5 and -.5 so toks get sorted in right order downstream
    labeled_toks = [{"word": "α", "index": -1000}] # alpha is SOS tag
    for counter, t in enumerate(toks):
        if t["index"] == mint:
            labeled_toks.append({"word": START, "index": t["index"] - .5})
        labeled_toks.append({"word": t["word"], "index": t["index"]})
        if t["index"] == maxt:
            labeled_toks.append({"word": END, "index": t["index"] + .5})
    labeled_toks = labeled_toks + [{"word": "Ω", "index": 1000}] # omega is EOS tag
    labeled_toks.sort(key=lambda x: float(x["index"]))
    return labeled_toks


def get_labeled_toks_revised(node, state_jdoc, op_proposed, original_jdoc):
    if len(state_jdoc["tokens"]) == 0:
        labeled_toks = [{"word": "α", "index": -1000},{"word": "Ω", "index":1000}]
        return labeled_toks

    dep = [_["dep"] for _ in state_jdoc["basicDependencies"] if _["dependent"] == node][0]

    #learning curve seems to be capping at 4M examples. but if you add more, uncomment this line to fill any unseen deps before rerunning.    
    #with open("preproc/ud.txt", "a") as of:
    #    of.write(dep + "\n")

    START = "β" + ud2symbols[dep] + op_proposed   # BETA is BracketStart
    END = "γ" + ud2symbols[dep] + op_proposed  # gamma is BracketEnd
    
    cut = dfs(g=state_jdoc, hop_s=node, D=[])
    cut.sort()
    mint = min(cut)
    maxt = max(cut)

    V = [i["index"] for i in state_jdoc["tokens"]]

    labeled_toks = [{"word": i["word"] + "τ", "index":i["index"]}
                    for i in original_jdoc["tokens"] if i["index"] not in V]

    labeled_toks = labeled_toks + [{"word": i["word"], "index":i["index"]}
                                   for i in original_jdoc["tokens"]
                                   if i["index"] in V]


    # if the proposed op is extract then the proposed tokens are NOT in compression
    if op_proposed == "ε":  # i.e. extract
        for vno, v in labeled_toks:
            if v["index"] in cut:
                labeled_toks[vno]["word"] = labeled_toks[vno]["word"] + "τ"

    # the indexes are added w/ +.5 and -.5 so toks get sorted in right order downstream

    # alpha is SOS tag for compression
    labeled_toks = labeled_toks + [{"word": "α", "index": min(V) - .5}]

    # δ is start of entire sequence
    labeled_toks = labeled_toks + [{"word": "δ", "index": -10000}]

    # λ is end of entire sequence
    labeled_toks = labeled_toks + [{"word": "λ", "index": 100000}]

    # omega is EOS tag for compression
    labeled_toks = labeled_toks + [{"word": "Ω", "index": max(V) + .5}]

    # add bracket tags
    labeled_toks.append({"word": START, "index": mint - .5})

    labeled_toks.append({"word": END, "index": mint + .5})

    labeled_toks.sort(key=lambda x: float(x["index"]))

    return labeled_toks


def prune_deletes_q(vertex, jdoc):
    '''would pruning this vertex delete any query items?'''
    q = jdoc["q"]
    pruned = dfs(g=jdoc, hop_s=vertex, D=[])
    return len(set(pruned) & set(q)) > 0


def get_len_mc(mc_indexes, jdoc):
    out = " ".join([_["word"] for _ in jdoc["tokens"] if _["index"] in mc_indexes])
    return len(out)

