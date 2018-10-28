import re


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