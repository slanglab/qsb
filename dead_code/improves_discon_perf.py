
#######
# This improves global perf. but you can't really justify only going global feats
#####


try:
    featsg['middle_dep'] = str(featsg['middle']) + feats[depf]
    featsg['right_add_dep'] = str(featsg['right_add']) + feats[depf]
    featsg["left_add_dep"] = str(featsg['left_add']) + feats[depf]
except KeyError:
    pass


####

#This inclues perf but there is not really a ton of reason to include this. The F and A features are for EDGES but there is no edge here

try:
    dep = [de for de in sentence['basicDependencies'] if de["governor"] == vertex][0]
    feats1 = get_features_of_dep(dep, sentence, depths)
    for f in feats1:
        feats["dg" + f] = feats1[f]
except IndexError:# root vx
    pass

try:
    dep = [de for de in sentence['basicDependencies'] if de["dependent"] == vertex][0]
    feats1 = get_features_of_dep(dep, sentence, depths)
    for f in feats1:
        feats["dgl" + f] = feats1[f]
except IndexError:# leaf vx
    pass


####################################################################
# Feature here w/ disconnected vertexes. It is cut in this case
####################################################################

def gov_of_proposed_is_verb_and_current_compression_no_verb(sentence, vertex, current_compression):
    if gov_is_verb(vertex, sentence):
        if not current_compression_has_verb(sentence=sentence,
                                            current_compression=current_compression):
            return True
    return False

verby = gov_of_proposed_is_verb_and_current_compression_no_verb(sentence,
                                                                vertex,
                                                                current_compression)
feats["proposed_governed_by_verb"] = verby


#### this imporoves disconnected perf


feats["gov_is_root"] = governor == 0

feats["is_next_tok"] = vertex == max(current_compression) + 1

if (vertex + 1 in current_compression and vertex - 1 in current_compression):
    feats["is_missing"] = 1
else:
    feats["is_missing"] = 0

