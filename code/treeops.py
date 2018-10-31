
'''
Various tree operations ...
'''
import copy
import random
from queue import Queue
import collections

DEPS = "basicDependencies"
ROOT = 0
PRUNE = 'p'
EXTRACT = 'e'
r = random.random()

# A subtree rooted at root s.t. if you do prune ops at pruned u get compression
Subtree = collections.namedtuple('Subtree', 'root pruned')


def has_dep(deps, kind):
    '''is there a dep of type kind?'''
    return any(d["dep"] == kind for d in deps)


def get_kind(deps, kind):
    '''get a dep of type kind. assumes that there is one b/c has_dep has been called'''
    nmods = [d for d in deps if d["dep"] == kind]
    random.shuffle(nmods, lambda: r)
    return nmods[0]


def get_kind_with_dep(deps, kind, dependent):
    '''
    - Get a dep of type kind with dependent = dependent
    - This is same as above, but there is no randomness in picking the dep
    '''
    nmods = [d for d in deps if d["dep"] == kind and d["dependent"] == dependent]
    return nmods[0]


def do_ops(ops, jdoc_sent, location = None):
    '''
    execute a list of ops of a jdoc_sent, e.g. [p-nmod ...]

    '''
    clone = copy.deepcopy(jdoc_sent)
    for o in ops:
        kind, label = o.split("-")
        if kind == PRUNE and has_dep(clone[DEPS], label):
            prune_here = get_kind(clone[DEPS], label)["dependent"]
            prune(clone, prune_here)
        elif kind == EXTRACT and has_dep(clone[DEPS], label):
            extract_here = get_kind(clone[DEPS], label)["dependent"]
            extract(clone, extract_here)
    return clone


def trigram_no_cut(s1, indexes_cut):
    '''
    - what is the trigram which would have been there had the cut not occured?
    - indexes_cut are the indexes which are removed from s1
    - e.g. I went to (school yesterday) with Joe, where (school yesterday)
        is cut should return "to school yesterday".
    '''
    cut = [i["word"] for i in s1["tokens"] if i["index"] == indexes_cut[0]][0]
    try:
        b1 = [i["word"] for i in s1["tokens"] if i["index"] == (indexes_cut[0] -1)][0]
    except:
        b1 = ""
    try:
        b2 = [i["word"] for i in s1["tokens"] if i["index"] == (indexes_cut[0] -2)][0]
    except:
        b2 = ""
    return " ".join([b2, b1, cut])


def trigram_made_by_cut(jdoc_sent, indexes_cut):
    '''
    what is the trigram created by performing the cut @ indexes_cut?
    returns a space-delimited trigram, probably for input to query a
    KLM lang model
    '''
    start_cut = min(indexes_cut)
    end_cut = max(indexes_cut)

    def get_tok_by_ix(jdoc_sent, ix):
        return [i for i in jdoc_sent["tokens"] if i['index'] == ix][0]

    retained = [i["word"] for i in jdoc_sent["tokens"] if i["index"] not in indexes_cut][0:-1] # cut off punct
    max_index_in_sent = max([i["index"] for i in jdoc_sent["tokens"]])
    min_index_in_sent = min([i["index"] for i in jdoc_sent["tokens"]])
    # there are three possible cases...

    if start_cut == min_index_in_sent:
        return "SOS " + " ".join([i for i in retained[0:2]])
    # -1 cus punctuation
    elif end_cut == max_index_in_sent -1 or end_cut == max_index_in_sent:
        return " ".join([i for i in retained[-2:]]) + " EOS"
    else:
        # in very rare cases, only the first word is retained from the start of a
        # sentence. hence this try/except block
        try:
            wn2 = get_tok_by_ix(jdoc_sent, start_cut - 2)['word']
        except:
            wn2 = ""
        wn1 = get_tok_by_ix(jdoc_sent, start_cut - 1)['word']
        try:
            wn0 = get_tok_by_ix(jdoc_sent, end_cut + 1)['word']
        except:
            return "EOS"
        return " ".join([wn2, wn1, wn0])


def subtrees_to_ctoks(source, subtrees):
    '''
    take a list of subtree named tuples and return indexes that remain
    once you execute the extract and prune ops encoded in the Subtree
    '''
    c = []
    for subtree in subtrees:
        st = copy.deepcopy(source)
        extract(st, subtree.root)
        for p in subtree.pruned:
            prune(st,p)
        c = c + [i['index'] for i in st['tokens']]
    return c


def subtrees_to_opts(source, subtrees):
    '''
    take a list of subtree named tuples and return indexes that remain
    once you execute the extract and prune ops encoded in the Subtree
    '''

    def get_incoming(ix):
        return [o["dep"] for o in source[DEPS] if o["dependent"] == ix][0]

    opts =[]
    for subtree in subtrees:
        incoming = get_incoming(subtree.root)
        e_op = "e-" + incoming
        p_ops = []
        for p in subtree.pruned:
            incoming = get_incoming(p)
            p_ops.append("p-" + incoming)
        opts.append({"e":e_op, "p":p_ops})
    return opts


def get_tree_if_has_compression(source, start_vx, compresion_ixs):
    '''
    Is it possible to find a non empty subtree rooted at
    start_vx which contains only nodes in compression IX?

    - returns Y/N answer, a list of chops to construct the tree, the tree
    '''
    d, pi, c = bfs(source,start_vx)
    D = dfs(source, start_vx, D=[])
    route = walk_tree(d)

    chops = []
    pruned_already = []
    tree = []


    for v in route:
        if v not in pruned_already:
            D = dfs(source, v, D = [])
            if v not in compresion_ixs:
                chops.append(v) # v is choppable
                pruned_already = pruned_already + D
            else:
                tree.append(v)

    success = all(t in compresion_ixs for t in tree)
    if len(tree) == 0:
        success = False

    return success, chops, tree


def find_maximal_subtrees(source, compresion_ixs):
    '''
    Start searching at root. This should return the maxmial subtrees s.t. all
    nodes in tree are in compression. Maximal b.c they are searched root down
    i.e. if t1 is in t2 then it will just return t2. maxmial is important so you
    don't double count.

    - We don't actually return the subtrees, just the root and list of vertexes
    - which can be pruned to create the subtree.
    '''
    START_VX = 0
    d, pi, c = bfs(source, START_VX)
    route = walk_tree(d) # will not return unreachable nodes ....
    trees = []
    pruned_already = []
    assert all(type(i) == int for i in compresion_ixs)
    subtrees = [] # stores a list of what prune opts created what trees
    for v in route:
        if v not in pruned_already:
            success, chop, tree = get_tree_if_has_compression(source, v, compresion_ixs)
            if success and len(tree) > 0:
                pruned_already = pruned_already + tree
                trees.append(tree)
                subtrees.append(Subtree(root=v, pruned=chop))
    return subtrees


def walk_tree(d):
    '''
    return a list of vertexes to walk, root to trees. d comes from BFS
    '''
    vertexes_and_depts = [(vertex,depth) for vertex,depth in d.items()]
    vertexes_and_depts.sort(key=lambda x:x[1])
    # -1 is the indicator that a vertex is not reachable
    return [o[0] for o in vertexes_and_depts if o[1] > -1]


def bfs(g, hop_s):
    '''
    breadth first search

    Args:
        g: a graph
        hop_s: integer starting vertex, our case a root (i.e. 0)

    Returns:
        - color list of found nodes, c
        - list of nodes and predecessor, pi (predecessor == parent if tree)
        - list of depths, d
    '''
    q = Queue()

    # ancestors
    pi = {i["dependent"]: None for i in g[DEPS]}
    # dependents
    d = {i["dependent"]: -1 for i in g[DEPS]}
    # colors
    c = {i["dependent"]: "W" for i in g[DEPS]}

    c[hop_s] = "G"
    d[hop_s] = 0

    q.put(hop_s)

    while not q.empty():
        u = q.get()
        for v in [i["dependent"] for i in g[DEPS] if i["governor"] == u]:
            if c[v] == "W":
                c[v] == "G"
                d[v] = d[u] + 1
                pi[v] = u
                q.put(v)
        c[u] = "B"

    return d, pi, c


def adj(g, ix):
    return [o["dependent"] for o in g[DEPS] if o["governor"] == ix]


def dfs(g, hop_s, D=[]):
    '''
    depth first search

    - find_new_adjacents is a call back that can be modified

    Args:
        g: a graph (i.e. a jdoc_sent for us)
        hop_s: starting vertex

    Returns:
        a list of discovered vertexes
    '''
    D.append(hop_s)
    for v_prime in adj(g, hop_s):
        if v_prime not in D:
            dfs(g, v_prime, D)
    return D


def prune(g, v):
    '''
    remove v and all of its descendants from g
    - pass by reference
    '''
    D = dfs(g, v, D =[])
    g["tokens"] = [t for t in g["tokens"] if t["index"] not in D]
    g[DEPS] = [d for d in g[DEPS] if d["governor"] not in D and d["dependent"] not in D]


def extract(g, v):
    '''
    get the subtree rooted at D
    - pass by reference
    '''
    D = dfs(g, v, D =[])
    g["tokens"] = [t for t in g["tokens"] if t["index"] in D]
    g[DEPS] = [d for d in g[DEPS] if d["governor"] in D and d["dependent"] in D]


def get_walk_from_root(jdoc):
    '''
    just a convenience method that returns a walk from root, breath-first
    '''
    d, pi, c = bfs(jdoc, 0)
    vertexes_and_depts = [(vertex,depth) for vertex,depth in d.items()]
    vertexes_and_depts.sort(key=lambda x:x[1])
    return [vertex for vertex, depth in vertexes_and_depts if depth > 0]
