# coding: utf-8
'''
This code shows how to reconstruct all oracle summaries based on 
prune and extract alone. Prune and extract are the reverse of 
each other. 

Buffer == a list of tokens
Extract == add this token and all of its children to the buffer
Prune == remove this token and all of its children from the buffer
'''

import json
from tqdm import tqdm
from code.treeops import bfs
from code.treeops import dfs
from code.treeops import walk_tree

lns = []

with open("preproc/training.jsonl", "r") as inf:
    for ino, _ in tqdm(enumerate(inf)):
        ln = json.loads(_)
        buffer_ = []

        d, pi, c = bfs(ln, 0)

        for _ in walk_tree(d):
            if _ != 0:
                oracle = ln["oracle"][str(_)]
                children = dfs(g=ln, hop_s=_, D=[])
                if oracle == "e":
                    buffer_ = buffer_ + children
                if oracle == "p":
                    for c in children:
                        if c in buffer_:
                            buffer_.remove(c)
        buffer_.sort()
        assert buffer_ == ln["compression_indexes"]
