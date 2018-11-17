# coding: utf-8
import json
with open("preproc/lstm_train_3way.jsonl", "r") as inf:
    for l in inf:
        l = json.loads(l)
        
before, after = [ino for ino,i in enumerate(l["tokens"]) if "OOV" in i["word"] and "proposed" in i["word"]]

toks = l["tokens"]
toks.sort(key=lambda x:x["index"])

vl = toks[0:before]
vr = toks[after+1:]
tv = toks[before + 1:after]
bracket1 = [toks[before]]
bracket2 = [toks[after]]

