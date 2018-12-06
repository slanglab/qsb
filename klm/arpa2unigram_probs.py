# http://www.speech.sri.com/projects/srilm/manpages/ngram-format.5.html

import json

with open("fa.arpa", "r") as inf:
    lns = [o for o in inf]
    unigram_probs = []
    started = False
    for ino, i in enumerate(lns):
        if started:
            unigram_probs.append(i.split("\t")[0:2])
        if '2-gr' in i:
            break
        if "1-gr" in i:
            started = True

    unigram_probs = [o for o in unigram_probs if len(o) == 2]
    unigram_probs = {o[1]: o[0] for o in unigram_probs}

with open("fa.unigrams.json", "w") as of:
    json.dump(unigram_probs, of)
