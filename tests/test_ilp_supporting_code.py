import json
import pytest

from ilp2008.util import jdoc_to_constit_list
from ilp2008.supporting_code import get_verb_ix, get_PPs, to_lists, get_SBARs, get_verb_groups
from ilp2008.supporting_code import get_coordination, get_personal_pronouns, to_lists
from ilp2008.supporting_code import get_det_mods, default_to_regular, get_alpha_scores
from ilp2008.supporting_code import get_possessives, get_negations, get_case, get_PPs
from code.printers import pretty_print_conl
from ilp2008.supporting_code import enum_alpha, generate_beta_indexes, enum_gamma, to_lists
from ilp2008.supporting_code import get_beta_scores, get_gamma_scores, get_SBARs
from ilp2008.supporting_code import load_sentence


FIXTURE_DIRECTORY = "tests/fixtures/"


def test_jdoc_to_constit_list():
    '''a constituent parse should be convertable into a list'''

    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    both_lists = type(jdoc_to_constit_list(jdoc)) == list
    assert both_lists, "constit_parse_to_list should return a list"


def test_verb_indexes():
    '''check the function which gets verb indexes'''
    pakistan = load_sentence(FIXTURE_DIRECTORY + "pakistan_example.json")
    total_verbs_in_sentence = len(get_verb_ix(pakistan))
    assert total_verbs_in_sentence == 3, "this sentence has 3 verbs"

    if __name__ == "__main__":
        for verb in get_verb_ix(pakistan):
            word = [i for i in pakistan["tokens"] if i["index"] == verb][0]
            print(word["word"])


def test_get_coordination():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    for i in get_coordination(jdoc):
        coordinated = [_ for _ in jdoc["tokens"] if _["index"] in i.values()]
        coordinated.sort(key=lambda c:c["index"])
        coordinated = " ".join([_["word"] for _ in coordinated])
        assert coordinated == "Grenada and Nicaragua", "Grenada and Nicaragua are coordinated in this sentence"


def test_jdoc_to_constit_list():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    assert type(jdoc_to_constit_list(jdoc)) == list


def test_personal_pronouns():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "swimming.txt.json")
    pronouns = get_personal_pronouns(jdoc)
    prounouns = [_["word"] for _ in jdoc["tokens"] if _["index"] in pronouns]
    assert len(pronouns) == 2, "there are 2 prounouns in the sentence"
    assert "He" in prounouns, "one of the prounouns is He"
    assert "you" in prounouns, "one of the prounouns is you"


def test_determiners():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    determiner_pairs = get_det_mods(jdoc)
    print(len(determiner_pairs))
    assert len(determiner_pairs) == 2, "there are two determiner pairs in this sentence"
    for dep, gov in determiner_pairs:
        t1,t2 = list(filter(lambda x:x["index"] in [dep, gov], jdoc["tokens"]))
        assert t1["word"] in ["The", "the"], "in this sentence, there are two determiners that are 'the'"


def test_default_to_reg():
    from collections import defaultdict
    d = defaultdict()
    d["3"] = 4
    assert type(default_to_regular(d)) == dict


def test_get_importance_scores():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    words = [str(i["word"]) for i in jdoc["tokens"]]
    scores = get_alpha_scores(words)
    assert len(scores) == len(jdoc["tokens"]), "each token should have an importance score"


def test_verb_groups():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "basic_verb.txt.json")
    for group in get_verb_groups(jdoc):
        assert len(group) == 2, "this should be a group of size 2"
        indexes = [group["verb"]] + group["args"]
        group = [_["word"] for _ in jdoc["tokens"] if _["index"] in indexes]
        assert "Bob" in group and "gave" in group and "cake" in group


def test_unpack_lists():
    all_lists = to_lists([[1, [2, 3], [3,[4,[5]]]]])
    assert len(all_lists) == 5, "there should be 5 nested lists, there are {}".format(len(all_lists))


def test_possessives():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "simple_possessive.txt.json")
    for group in get_possessives(jdoc):
        group = [_["word"] for _ in jdoc["tokens"] if _["index"] in group]
        # Jeff 's book
        assert len(group) == 3 and "'s" in group and "book" in group and "Jeff" in group


def test_negation():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "simple_negation.txt.json")
    for group in get_negations(jdoc):
        assert len(group) == 3
        group = [_["word"] for _ in jdoc["tokens"] if _["index"] in group]
        assert "go" in group and 'did' in group and "n't" in group


def test_negation2():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "simple_negation2.txt.json")
    # Bob is not safe.
    for group in get_negations(jdoc):
        words = [_["word"] for _ in jdoc["tokens"] if _["index"] in group]
        assert words == ["not", "safe"]


def test_case_edges():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "simple_possessive.txt.json")
    for group in get_case(jdoc):
        group = [_["word"] for _ in jdoc["tokens"] if _["index"] in group]
        # Jeff 's
        assert len(group) == 2, "should be a group of length 2"


def test_PPs():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    if __name__ == "__main__":
        for i in get_PPs(jdoc):
            print([_["word"] for _ in jdoc["tokens"] if _["index"] in i["children"]])
    assert len(list(get_PPs(jdoc))) == 3


def to_words(jdoc):
    for w in jdoc['tokens']:
        yield w["word"]


def test_enum_alpha_indexing():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    alphas = enum_alpha(list(to_words(jdoc))) #alpha number -> word
    assert 0 not in alphas, "alphas are indexed at 1"


def test_enum_alpha_type():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    alphas = enum_alpha(list(to_words(jdoc))) #alpha number -> word
    assert type(alphas) == dict, "alphas are a dict"


def test_enum_alpha_values():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    alpha_values = enum_alpha(list(to_words(jdoc))).values()
    assert all(i["word"] in alpha_values for i in jdoc["tokens"])


def test_generate_beta_indexes_indexing():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    betas = generate_beta_indexes(list(to_words(jdoc))) #alpha number -> word
    assert 0 in betas, "first level of beta dict are indexed at 1"


def test_generate_beta_indexes_indexing_2():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    betas = generate_beta_indexes(list(to_words(jdoc))) #alpha number -> word
    for b in betas:
        assert 0 not in betas[b], "second level of beta dict are indexed at 1"
    for i in betas:
        assert i + 1 == min(betas[i]), "second level of beta dict are indexed at i"


def test_enum_gamma_type():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    gammas = enum_gamma(list(to_words(jdoc))) #alpha number -> word
    n = len(jdoc["tokens"])
    for b in gammas:
        assert 0 not in gammas[b], "second level of gammas dict are indexed at 1"
    for i in gammas:
        assert i + 1 == min(gammas[i]), "second level of beta dict are indexed at i"
        for j in gammas[i]:
            assert j + 1 == min(gammas[i][j]), "second level of beta dict are indexed at i"
            assert max(gammas[i][j]) == n


def test_get_beta_scores():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    scores = get_beta_scores(list(to_words(jdoc))) #alpha number -> word
    for i in scores:
        for j in scores[i]:
            assert scores[i][j] < 0, "score should be a log prob from a LM"


def test_get_gamma_scores():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "uganda_example.txt.json")
    scores = get_gamma_scores(list(to_words(jdoc))) #alpha number -> word
    for i in scores:
        for j in scores[i]:
            for k in scores[i][j]:
                assert scores[i][j][k] < 0, "score should be a log prob from a LM"


def test_sbars():
    jdoc = load_sentence(FIXTURE_DIRECTORY + "simple_sbar.txt.json")
    assert len(list(get_SBARs(jdoc))) > 0, "there should be a SBAR in the list"
    if __name__ == "__main__":
        for s in get_SBARs(jdoc):
            print([_["word"] for _ in jdoc["tokens"] if _["index"] in s['children']])
