import pytest
import scipy
import json

from code.printers import *
from ilp2013.fillipova_altun_supporting_code import *


def load_sentence(filename):
    with open(filename, "r") as inf:
        jdoc = json.load(inf)["sentences"][0]
        return filippova_tree_transform(jdoc)

all_vocabs = get_all_vocab_quick_for_tests()

namer = FeatureNamer(vocabs=all_vocabs)

def test_get_tok():
    sent = load_sentence("tests/fixtures/basic_verb.txt.json")
    assert type(get_tok(6, sent)) == dict
    assert get_tok(6, sent)['word'] == "Alice"

def test_get_siblings():
    sent = load_sentence("tests/fixtures/basic_verb.txt.json")
    e = (2,6)
    deps = [i for i in get_siblings(e, sent)]
    assert len(deps) == 3, "cake has 3 non punctuation siblings"

test_get_siblings()

def test_get_edge_type():
    sent = load_sentence("tests/fixtures/basic_verb.txt.json")
    e = (2,6)
    e = get_edge(2,6, sent)
    assert type(e) == dict

def test_label_type_and_value():
    sent = load_sentence("tests/fixtures/basic_verb.txt.json")
    e = (2,6)
    label_ = label(2,6, sent)
    assert type(label_) == unicode
    assert label_[0:4] == "nmod"

def test_is_negated():
    sent = load_sentence("tests/fixtures/simple_negation.txt.json")
    assert is_negated(4, sent) == True, "the word go is negated"
    assert is_negated(6, sent) == False, "the word Texas is not negated"

def test_is_negated2():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    assert is_negated(4, sent) == True, "the word safe is negated"
    assert is_negated(1, sent) == False, "the word Bob is not negated"

def test_pos():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    print pos(2, sent)
    assert pos(2, sent) == "VBZ", "token 2, is, is negated"
    assert lemma(2, sent) == "be", "the word is has lemma be"

def test_ne():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    assert ne_tag(2, sent) == "O"

def test_num_children():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    pretty_print_conl(sent)
    assert num_children(4, sent) == 3, "the word safe has 3 non punctuation children"


def test_char_len():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    assert char_length(4, sent) == 4, "the word safe has 4 chars"

def test_no_words_in():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    no_words_in(4, sent)

def test_depth():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    d, pi, c = bfs(sent, hop_s=0)
    assert depth(index=2, d=d) == 2
    assert depth(index=4, d=d) == 1

def test_syntacic_runs():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    out = syntactic((4,3), sent, all_vocabs)
    assert type(out) == scipy.sparse.csr.csr_matrix
    assert np.sum(out.toarray()) > 0 and np.sum(out.toarray()) < 4

def test_sematic_runs():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    out = semantic((4,3), sent, all_vocabs)
    assert type(out) == scipy.sparse.csr.csr_matrix
    assert np.sum(out.toarray()) > 0 and np.sum(out.toarray()) < 4

def test_structural_runs():
    sent = load_sentence("tests/fixtures/simple_negation2.txt.json")
    out = structural(e=(4,3), jdoc=sent)
    assert type(out) == scipy.sparse.csr.csr_matrix

def test_lexical_runs():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    out = lexical(e=(4,3), jdoc=jdoc, vocabs=all_vocabs)
    assert type(out) == scipy.sparse.csr.csr_matrix

def test_f_runs():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    e = (4,3)
    out = f(e=e, jdoc=jdoc, vocabs=all_vocabs)
    assert type(out) == scipy.sparse.coo.coo_matrix

def test_lexical_names_same_size_as_namer():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    lexical_out = lexical(e=(4,3), jdoc=jdoc, vocabs=all_vocabs)
    assert len(namer.lexical_names) == lexical_out.shape[1]

def test_structural_names_same_size_as_namer():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    structural_out = structural(e=(4,3), jdoc=jdoc)
    assert len(namer.structural_names) == structural_out.shape[1]

def test_semantic_names_same_size_as_namer():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    out = semantic((4,3), jdoc, all_vocabs)
    assert len(namer.semantic_names) == len(out)

def test_semantic_names_same_size_as_namer():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    out = syntactic((4,3), jdoc, all_vocabs)
    assert len(namer.syntactic_names) == out.shape[1]


def test_lexical_names_correctness():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    h, n = 4,3
    lexical_out = lexical(e=(h, n), jdoc=jdoc, vocabs=all_vocabs)
    lookup_ix = namer.lexical_names.index("lemma_n:" + lemma(jdoc=jdoc, index=n))
    print "todo"

def test_random_scores_runs():
    jdoc = load_sentence("tests/fixtures/simple_negation2.txt.json")
    scores = get_random_dependency_scores(jdoc)
    for d in jdoc["enhancedDependencies"]:
        assert type(scores["{}-{}".format(d["governor"], d["dependent"])]) == float

def test_in_A_but_not_B():
    A = set([1,2,3])
    B = set([2,3])
    assert len(A_but_not_B(A,B)) == 1
    assert [_ for _ in A_but_not_B(A,B)][0] == 1
