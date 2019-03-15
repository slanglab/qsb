'''
This file is a bridge between regular preprocessing and preprocessing for AllenNLP
'''

from tqdm import tqdm
import glob
import json 

WORD_TOKEN_DELIMITER="\t"

def pickled_preproc_to_string(filename):
    '''
    inputs:
        filename(str): the name of a pickled file of jdoc sentences

    This method writes this file in the format shown below for AllenNLP
    '''

    def to_string_and_tag(jdoc):
        compression_indexes = jdoc["compression_indexes"]
        def to_tag(tok):
            if tok["index"] == max(compression_indexes):
                return "3" # EOS tag, numeric code
            if tok["index"] == min(compression_indexes):
                return "2" # SOS tag, numeric code
            return "1" if tok["index"] in compression_indexes else "0"
        return WORD_TOKEN_DELIMITER.join([_["word"] + "###" + to_tag(_) for _ in jdoc['tokens']])

    with open(filename.replace(".jsonl", ".txt").replace("preproc/", "data/"), "w") as of:
        with open(filename, "r") as inf:
            for _ in tqdm(inf):
                dt = to_string_and_tag(json.loads(_))
                of.write(dt + "\n")


    with open(filename.replace(".jsonl", ".forpreds").replace("preproc/", "data/"), "w") as of:
        with open(filename, "r") as inf:
            for line_ in tqdm(inf):
                dt = {"sentence": WORD_TOKEN_DELIMITER.join([_["word"] for _ in json.loads(line_)["tokens"]]), "jdoc": line_}
                of.write(json.dumps(dt) + "\n")

if __name__ == "__main__":
    for _ in glob.glob("preproc/*jsonl"):
        pickled_preproc_to_string(filename=_)
