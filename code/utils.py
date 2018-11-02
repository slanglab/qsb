import re
from code.treeops import dfs
from code.treeops import bfs

def ner_to_s(tok):
    if tok["ner"] == "PERSON":
        return "P"
    if tok['ner'] == "LOCATION":
        return "L"
    if tok["ner"] == "ORGANIZATION":
        return "O"
    return "X"


def get_ner_string(jdoc):
    '''make a big string showing NER tag at each position'''
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


def get_labeled_toks(node, jdoc):
    if len(jdoc["tokens"]) == 0:
        labeled_toks = [{"word": "SOS"},{"word": "EOS"}]
        return labeled_toks
    dep = [_["dep"] for _ in jdoc["basicDependencies"] if _["dependent"] == node][0]
    START = "OOVSTART" + dep
    END = "OOVEND" + dep
    toks = [i for i in jdoc["tokens"]]
    cut = dfs(g=jdoc, hop_s=node, D=[])
    cut.sort()
    mint = min(cut)
    maxt = max(cut)
    # This assertion is false in cases where you greedily prune in the middle of 
    # trees. it is true if you only only prune a whole branch 
    #assert len(cut) == len(range(mint, maxt + 1))
    labeled_toks = [{"word": "SOS"}]
    for counter, t in enumerate(toks):
        if t["index"] == mint:
            labeled_toks.append({"word": START, "index": t["index"]})
        labeled_toks.append({"word": t["word"], "index": t["index"]})
        if t["index"] == maxt:
            labeled_toks.append({"word": END, "index": t["index"]})
    return labeled_toks + [{"word": "EOS"}]


def prune_deletes_q(vertex, jdoc):
    '''would pruning this vertex delete any query items?'''
    q = jdoc["q"]
    pruned = dfs(g=jdoc, hop_s=vertex, D=[])
    return len(set(pruned) & set(q)) > 0
