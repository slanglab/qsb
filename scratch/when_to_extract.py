from __future__ import print_function

import json

from tqdm import tqdm
from code.utils import is_prune_only
from code.utils import get_min_compression

if __name__ == "__main__":

    with open("preproc/validation.jsonl", "r") as inf:
        for vno, _ in tqdm(enumerate(inf)):
            jdoc = json.loads(_)
            r = int(jdoc["r"])
            po = is_prune_only(jdoc)
            Q = jdoc["q"]
            mc = [get_min_compression(v, jdoc) for v in Q]
            mc = [i for m in mc for i in m]
            mcl = " ".join([o["word"] for o in
                            jdoc["tokens"] if o["index"] in mc])

            print(r, po, mcl)
