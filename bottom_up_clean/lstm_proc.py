import json


C_char = "π"  # "proposedprune"
v_char = "ε"  # 'proposedextract'

f_char = "F"




def get_sorted_toks_with_markers(sentence, C, v, F):
    C_toks = [(_["word"] + C_char, _["index"]) for _ in sentence["tokens"] if _["index"] in C]
    V_toks = [(_["word"] + v_char, _["index"]) for _ in sentence["tokens"] if _["index"] == v]
    F_toks = [(_["word"] + f_char, _["index"]) for _ in sentence["tokens"] if _["index"] in F and _["index"] != v]
    S_toks = [(_["word"] + f_char, _["index"]) for _ in sentence["tokens"] if _["index"] not in C + F + [v]]

    tok = C_toks + V_toks + F_toks + S_toks
    tok.sort(key=lambda x:x[1])

    # assert len(set(C_toks + V_toks + F_toks + S_toks)) == len(sentence["tokens"])

    return [_[0] for _ in tok]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', type=str, dest="fn")

    args = parser.parse_args()

    fn = args.fn

    tx = [json.loads(i) for i in open(fn)]

    with open(fn.replace(".paths", ".lstm.jsonl"), "w") as of:
        for i in tx:

            paths, sentence = i["paths"], i["sentence"]

            for p in paths:

                C, v, decision, F = p

                toks = get_sorted_toks_with_markers(sentence, C, v, F)            

                label = decision

                out = {"sentence": " ".join(toks), "label": str(label)}
                of.write(json.dumps(out) + "\n")