
def test_tree_ops():
    from code.treeops import bfs
    import json

    with open("tests/fixtures/swimming.txt.json", "r") as inf:
        dt = json.load(inf)["sentences"][0]

    d,c,p = bfs(g=dt, hop_s=0)        
    assert d[7] == 3
    assert d[7] > c[1]
