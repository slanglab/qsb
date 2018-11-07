# coding: utf-8
'''
This code shows how to reconstruct all oracle summaries based on 
prune and extract alone. Prune and extract are the reverse of 
each other. 

Buffer == a list of tokens
Extract == add this token and all of its descendants to the buffer
Prune == remove this token and all of its descendants from the buffer

You need to do it BFS
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

        for vertex in walk_tree(d):  # get a breadth-first walk
            if vertex != 0:
                oracle = ln["oracle"][str(vertex)]
                descendants = dfs(g=ln, hop_s=vertex, D=[]) # vertex and descendants
                if oracle == "e":
                    buffer_ = buffer_ + descendants
                if oracle == "p":
                    for c in descendants:
                        if c in buffer_:
                            buffer_.remove(c)
        buffer_.sort()
        assert buffer_ == ln["compression_indexes"]
