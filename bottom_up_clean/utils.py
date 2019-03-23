from queue import Queue

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

    deps = "basicDependencies"

    # ancestors
    pi = {i["dependent"]: None for i in g[deps]}
    # dependents
    d = {i["dependent"]: -1 for i in g[deps]}
    # colors
    c = {i["dependent"]: "W" for i in g[deps]}

    c[hop_s] = "G"
    d[hop_s] = 0

    q.put(hop_s)

    while not q.empty():
        u = q.get()
        for v in [i["dependent"] for i in g[deps] if i["governor"] == u]:
            if c[v] == "W":
                c[v] == "G"
                d[v] = d[u] + 1
                pi[v] = u
                q.put(v)
        c[u] = "B"

    return d, pi, c