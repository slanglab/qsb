    # Here are some other features
    out["dependentGloss"] = dep["dependentGloss"]

    def get_children_deps(sentence, vertex):
        '''returns: count of children of vertex in the parse'''
        return [i["dep"] for i in sentence["basicDependencies"] if i["governor"] == vertex]

    def get_children_pos(sentence, vertex):
        '''returns: count of children of vertex in the parse'''
        dependents = [i["dependent"] for i in sentence["basicDependencies"] if i["governor"] == vertex]
        toks = [get_token_from_sentence(sentence, i) for i in dependents]
        return [o["pos"] for o in toks]

    for c in get_children_deps(sentence, dep["dependent"]):
        out[c + "_is_child"] = 1

    for c in get_children_pos(sentence, dep["dependent"]):
        out[c + "_pos_is_child"] = 1

    out["depth_dependent"] = depths[dep["dependent"]]
    out["position_dependent"] = float(dep["dependent"]/len(sentence["tokens"]))
    out["is_punct_dependent"] = dep["dependentGloss"] in PUNCT
    out["is_punct_gov"] = dep["governorGloss"] in PUNCT
    out["last2_dependent"] = dep["dependentGloss"][-2:]
    out["last2_gov"] = dep["governorGloss"][-2:]
    out["comes_first"] = dep["governor"] < dep["dependent"]
    out["governor_in_q"] = dep["governor"] in sentence["q"]
    out["gov_is_upper"] = dep["governorGloss"][0].isupper()
    out["dep_is_upper"] = dep["dependentGloss"][0].isupper()
    out["both_upper"] = out["gov_is_upper"] and out["dep_is_upper"] 

    # 0 means governor is root, usually if of the governing verb. Note flip of gov/dep in numerator
    if dep["governor"] == 0:
        out["position_governor"] = float(dep["dependent"]/len(sentence["tokens"]))
    else:
        out["position_governor"] = float(dep["governor"]/len(sentence["tokens"]))

    return dict(out)