'''
Make a "flat" file of every token in the corpus
'''

import glob
import json
from unidecode import unidecode
for fn in glob.glob("preproc/training.jsonl"):
    with open(fn, "r") as inf:
        try:
            for ln in inf:
                s = json.loads(ln)
                print 'SOS',
                for t in s["tokens"]:
                    print unidecode(t["word"]),
                print "EOS",
        except UnicodeError:
            pass 
