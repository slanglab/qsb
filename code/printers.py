'''
methods for printing things...
'''

import string
import re


def clean_vox_sentence(_str):
    _str = _str.replace("-LRB-", "(").replace("-RRB-", ")")
    _str = _str.replace("-LSB-", "[").replace("-RSB-", "]")


def strike(text):
    return ''.join([u'\u0336{}'.format(c) for c in text])


def strike_if_gone(tok, compressed_toks):
    '''
    - tok comes from the source sentence
    - compressed_toks is the compressed sentence's tokens
    '''
    ixs = [i["index"] for i in compressed_toks]
    if tok["index"] in ixs:
        return tok["word"]
    else:
        return strike(tok["word"])


def strike_if_gone_html(source_tok, compressed_toks):
    '''
    - source_tok is a token from the source sentence
    - compressed toks are the tokens in the compression
    - same as strike_if_gone, but returns BLANK STRINGS instead of crossouts
    '''
    ixs = [i["index"] for i in compressed_toks]
    if source_tok["index"] in ixs:
        return source_tok["word"]
    else:
        return "<strike>" + source_tok["word"] + "</strike>"


def print_striken(source, compression):
    compressed_toks = compression["tokens"]
    print(" ".join([strike_if_gone(t, compressed_toks) for t in source["tokens"]]))

def get_striken_html(source, compression):
    compressed_toks = compression["tokens"]
    return " ".join([strike_if_gone_html(t, compressed_toks) for t in source["tokens"]])


def pretty_print_conl(sentence, dependencies_kind = "basicDependencies"):
    '''pretty print deps in a conl-ish format'''
    aa = sentence[dependencies_kind]
    aa.sort(key=lambda x:x["dependent"])
    for a in aa:
        print("\t".join([str(a["dependent"]), a["dependentGloss"], "<-" + a["dep"] + "-", a["governorGloss"]]))


def dbp_hop(o):
    '''debug print a hop'''
    return "(" + o.note + ", " +str(o.label) + ")" if o.type == "W" else o.label + " " + o.note


def debug_print_path(path):
    '''debug print a path'''
    return " ".join([dbp_hop(o) for o in path])


def close_parens(str_):
    if "``" in str_ and '"' not in str_: # if there are only 1 parens
        str_ = str_.replace("``", "")

    if str_.count('"') == 1:  # if there are only one quote, close them
        str_.replace("''", "")

    return str_

def clean(out):
    '''
    compression can screw up punctuation.
    this tries a little bit of heuristic cleanup
    '''
    if out[0:2] == ", ":
        out = out[2:]

    out = out.replace("``", '"') # normalize quote characters

    out = out.replace("''", '"')

    out = out.replace("-LRB-", "(").replace("-RRB-", ")") # normalize brackets

    out = out.replace(" , ", ", ").replace(" .", ".").replace(" 's", "'s")
    out = out.replace(" ; ", "; ")
    out = out.replace("\\ ", "").replace("` ", "`").replace(" :", ":")

    # June 27 -- Fifteen-night voyage from Lisbon to Amsterdam, calling at La Coruna, Bilbao, St. Jean De-Luz, Bordeaux, , St. Malo, Rouen and Antwerp.
    out = out.replace(" , ,", ", ").replace(" ?", "?")
    out = out.replace("do n't", "don't") # quick fix some common contractions seen in data
    out = out.replace("does n't", "doesn't")
    out = out.replace("you 're", "you're")
    out = out.replace("wo n't", "won't")
    out = out.replace("they 'd", "they'd")
    out = out.replace("They 've", "They've")
    out = out.replace("they 've", "they've")
    out = out.replace("ca n't", "can't")
    out = out.replace("Do n't", "Don't")
    out = out.replace("They 're", "They're")
    out = out.replace("have n't", "haven't")
    out = out.replace(", , ", ", ").replace("Is n't", "Isn't")

    out = out.replace(" ) ", ") ").replace(" ( ", " (")

    if out.count('"') == 1: # sometimes prune messes up quotes. So heuristic fix
        out = out.replace('"', "")
    if out.count("''") == 1:
        out = out.replace("''", "")

    out = re.sub('\$ (?=[0-9])', '$', out)
    out = close_parens(out)

    out = out.replace(",.", ".")

    if len(out) > 0:

         # add a period if one is missing. Add caps if start ! caps
        if not out[0].isupper():
            out = out[0].upper() + out[1:]
        if not out[-1] in string.punctuation:
            out = out + "."

    out = out.replace(". .", ".")

    return out
