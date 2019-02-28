'''
Make a "flat" file of every token in the corpus
'''

import glob
import json
import sys

for fn in glob.glob("../preproc/training.jsonl"):
    with open(fn, "r") as inf:
        try:
            for ln in inf:
                out = ""
                s = json.loads(ln)
                for t in s["tokens"][:-1]: # xclude punct in LM. ILP and LSTM learn to exclude
                    out = out + " " + t["word"]
                print(out) 
        except UnicodeError:
            pass 
