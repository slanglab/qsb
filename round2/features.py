'''
features for binary logistic regression

Important:
    - CV features should end in _cv
        - e.g. f_pos_tag_of_determiner_parent_cv
    - CV features return lists
    - CV features must call myself() before each item in the list
    - f_g means a global feature (i.e. no interaction)
    - h_ means a helper function

'''
from __future__ import division
from code.treeops import bfs
from code.treeops import dfs
from code.log import logger
from round2 import BASEDIR
from round2 import COLLOCATION_DIRECTORY
from round2 import unigram_probs
from round2 import klm
from round2.greedy_summarizer import BVCompressor
from round2.predictors import FigureEightPredictor
import string
import inspect
import json
import copy
import math

PUNCT = [i for i in string.punctuation] + ["--"]

VERBOSE = False

COLLOCATION_VARIANCE_THRESHOLD = 2

RHS_COLLOCATION_THRESHOLD = 1.5

# https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback

myself = lambda: inspect.stack()[1][3] + ":"

cuv = get_cuv(cuv_file=BASEDIR + "/cuvplus/cuvplus.txt")


def f_is_nmod(d):
    return d["dep"] == "nmod"


def f_is_unigram_amod(d):
    '''e.g. Welcome  => welcome '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return d["dep"] == 'amod' and len(cut_tokens) == 1


def f_is_unigram_nsubj(d):
    '''e.g. Welcome  => welcome '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return d["dep"] == 'nsubj' and len(cut_tokens) == 1


def h_simulate_op_and_get_ix_affected(d):
    '''
    this is a helper method which returns the indexes cut from a given op

    for a prune these are actually cut
    for an extract these are actually retained
    '''
    u = set(dfs(d["source_json"], int(d["vertex"]), D=[])) | set([int(d["vertex"])])
    return [int(a) for a in u]


def h_simulate_op_and_get_ix_affected_multi_op(d):
    '''
    this is a helper method which returns the indexes cut from a given chain of ops.
    It is v. similar to the single op version

    for a prune these are actually cut
    for an extract these are actually retained
    '''
    out = []
    dv = d['vertex'].decode('string-escape').strip('"')
    for v in json.loads(dv):
        out = out + dfs(d["source_json"], int(v), D=[])
        out.append(v)
    u = set(out)
    return [int(a) for a in u]


def f_is_unigram_advmod(d):
    if d["dep"] == "advmod" and len(h_simulate_op_and_get_ix_affected(d)) == 1:
        return True
    else:
        return False


def f_get_advmod_unigram(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    word_cut = [_["word"] for _ in d["source_json"]["tokens"] if _["index"] in ix_cut][0] 
    if d["dep"] == "advmod" and len(h_simulate_op_and_get_ix_affected(d)) == 1:
        return [word_cut]
    else:
        return []


def f_amod_one_tok(d):
    if d["dep"] == "advmod" and len(h_simulate_op_and_get_ix_affected(d)) == 1: 
        return True
    else:
        return False


def f_last_pos_in_compression_cv(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    t = [myself() + _["pos"] for _ in d["source_json"]["tokens"] if _["index"] not in ix_cut and _["word"] not in PUNCT]
    return t


def f_is_nmod_and_starts_with_preposition(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] == "nmod":
        word_cut = [_["pos"] for _ in d["source_json"]["tokens"] if _["index"] in ix_cut][0] 
        if word_cut == "IN":
            return 1
        else:
            return 0 
    else:
        return 0


def f_not_starts_or_ends_sentence(d):
    if f_starts_sentence(d) or f_ends_sentence(d):
        return 0
    else:
        return True


def f_ends_sentence_case(d):
    if d["dep"] == "case":
        return f_ends_sentence(d)
    else:
        return False


def f_ends_sentence_xcomp(d):
    if d["dep"] == "xcomp":
        return f_ends_sentence(d)
    else:
        return False


def f_advmod_one_tok_ends_ly(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    ends_ly = [_["word"][-2:] for _ in d["source_json"]["tokens"] if _["index"] in ix_cut][0] == "ly" 
    if d["dep"] == "advmod" and len(h_simulate_op_and_get_ix_affected(d)) == 1 and ends_ly:
        return True
    else:
        return False


def f_advmod_one_tok_ends_y(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    ends_ly = [_["word"][-1:] for _ in d["source_json"]["tokens"] if _["index"] in ix_cut][0] == "y" 
    if d["dep"] == "advmod" and len(h_simulate_op_and_get_ix_affected(d)) == 1 and ends_ly:
        return True
    else:
        return False


def f_op_is_prune(d):
    out = 1 if d["op"] == "prune" else 0
    return out


def f_average_log_prob(d):
    '''
    average log prob of a sequence
    See http://www.aclweb.org/anthology/P14-2029
    '''
    s = d["source_json"]
    toks = ["SOS"] + [o["word"] for o in s["tokens"]] + ["EOS"]
    sw = " ".join(toks)
    return klm.score(sw)/len(s["tokens"])


def h_get_governor_token(d):
    '''get the token of the governor'''
    gov = [int(_["governor"]) for _ in 
           d["source_json"]["basicDependencies"] 
           if _["dependent"] == int(d["vertex"])][0]
    return [_["word"] for _ in d["source_json"]["tokens"] if int(_["index"]) == gov][0]


def f_ends_in_exclamation_point(d):
    '''source ends in exclamation?'''
    words = [_["word"] for _ in d["source_json"]["tokens"]]
    return words[-1] == "!"


def f_edge_is_case_and_unigram_cut_is_with(d):
    '''usually bad'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if len(cut_tokens) != 1: # if not exactly 1 token cut
        return False
    cut_words = [_["word"] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens][0]
    if cut_words[0] in ["with", "by"]: # if cut words != with
        return False
    return d["dep"] == "case"


MINP = min(unigram_probs.values())


def h_lookup(_):
    try:
        out = unigram_probs[_]
        return out
    except KeyError: # OOV. this is the min prob in the dict. 
        return MINP


def h_norm_lp(tokens):
    '''the normlp function from lau clarke and lapin'''
    p_m = klm.score(" ".join(tokens))
    p_u = sum([h_lookup(_) for _ in tokens])
    return -1 * p_m/p_u


def h_SLOR(tokens):
    '''
    The SLOR function on p. 1222 of Clark and Lapin

    inputs:
        tokens(list:string): a list of words
    '''
    p_m = klm.score(" ".join(tokens))

    p_u = sum([h_lookup(_) for _ in tokens])
    return (p_m - p_u)/len(tokens)


def f_g_clark_and_lapin_binary(d):
    '''
    features closely based on the the clarke and lapin work on acceptability
    adapted to compression setting
    inputs:
        d(dictionary): a jsonl row from crowdflower

    I guess there are a few interpretations of the Clark et al. features for our compression problem

    - one is just do SLOR on the compression. SLOR cor = .1228646
    - the other is do SLOR source - SLOR compression. This will get you the delta
        - this one is a little funny tho. what if your compression cuts off a fine but 
          somewhat improbable sequence? you dont want to get higher rank for that. I
          am just doing SLOR or the compression I think for now
    '''
    s = d["source_json"]
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    sw = ["SOS"] + [o["word"] for o in s["tokens"]] + ["EOS"]
    cw = ["SOS"] + [o["word"] for o in s["tokens"] if o["index"] not in cut_tokens] + ["EOS"]

    return (h_SLOR(cw) - h_SLOR(sw)) > 0


def f_g_clark_and_lapin_normlp_of_c(d):
    '''
    features closely based on the the clarke and lapin work on acceptability
    adapted to compression setting
    inputs:
        d(dictionary): a jsonl row from crowdflower

    I guess there are a few interpretations of the Clark et al. features for our compression problem

    - one is just do SLOR on the compression. SLOR cor = .1228646
    - the other is do SLOR source - SLOR compression. This will get you the delta
        - this one is a little funny tho. what if your compression cuts off a fine but 
          somewhat improbable sequence? you dont want to get higher rank for that. I
          am just doing SLOR or the compression I think for now
    '''
    s = d["source_json"]
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    cw = ["SOS"] + [o["word"] for o in s["tokens"] if o["index"] not in cut_tokens] + ["EOS"]

    return (h_norm_lp(cw))


def f_g_clark_and_lapin_normlp_binary(d):
    '''
    features closely based on the the clarke and lapin work on acceptability
    adapted to compression setting
    inputs:
        d(dictionary): a jsonl row from crowdflower

    I guess there are a few interpretations of the Clark et al. features for our compression problem

    - one is just do SLOR on the compression. SLOR cor = .1228646
    - the other is do SLOR source - SLOR compression. This will get you the delta
        - this one is a little funny tho. what if your compression cuts off a fine but 
          somewhat improbable sequence? you dont want to get higher rank for that. I
          am just doing SLOR or the compression I think for now
    '''
    s = d["source_json"]
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    sw = ["SOS"] + [o["word"] for o in s["tokens"]] + ["EOS"]
    cw = ["SOS"] + [o["word"] for o in s["tokens"] if o["index"] not in cut_tokens] + ["EOS"]

    return (h_norm_lp(cw) - h_norm_lp(sw)) > 0


def f_slor_of_c(d):
    '''
    features closely based on the the clarke and lapin work on acceptability
    adapted to compression setting
    inputs:
        d(dictionary): a jsonl row from crowdflower

    I guess there are a few interpretations of the Clark et al. features for our compression problem

    - one is just do SLOR on the compression. SLOR cor = .1228646
    - the other is do SLOR source - SLOR compression. This will get you the delta
        - this one is a little funny tho. what if your compression cuts off a fine but 
          somewhat improbable sequence? you dont want to get higher rank for that. I
          am just doing SLOR or the compression I think for now  
    '''
    s = d["source_json"]
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    cw = ["SOS"] + [o["word"] for o in s["tokens"] if o["index"] not in cut_tokens] + ["EOS"]

    return h_SLOR(cw)


def f_slor_of_c_xcomp(d):
    if d["dep"] == "xcomp":
        return f_slor_of_c(d)
    return 0


def f_slor_of_c_nsubj(d):
    if d["dep"] == "nsubj":
        return f_slor_of_c(d)
    return 0


def f_slor_of_c_aux(d):
    if d["dep"] == "aux":
        return f_slor_of_c(d)
    return 0


def f_ends_sentence(d):
    max_cut = max(h_simulate_op_and_get_ix_affected(d))
    indexes = [_["index"] for _ in d["source_json"]["tokens"] if _["index"] > max_cut and _["word"] not in PUNCT]
    return len(indexes) == 0


def f_ends_sentence_nsubj(d):
    if d['dep'] == "nsubj":
        return f_ends_sentence(d)
    return False


def f_is_nmod_and_is_short(d):
    '''
    e.g. He lived in Washington  => He lived

    Often this is a good feature to find yeses for nmod. usually these can be cut
    '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] == "nmod" and len(cut_tokens) < 5:
        return True
    return False


def f_starts_sentence(d):
    '''true if an op cuts tokens at the start of a sentence'''
    return min(h_simulate_op_and_get_ix_affected(d)) == 1


def f_starts_sentence_case(d):
    '''true if an op cuts tokens at the start of a sentence'''
    if d["dep"] == "case":
        return f_starts_sentence(d)
    else:
        return False


def f_starts_sentence_dep(d, target_dep):
    '''true if an op cuts tokens at the start of a sentence'''
    if d["dep"] == target_dep:
        return f_starts_sentence(d)
    else:
        return False


def f_one_word_case_prep(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if len(cut_tokens) == 1:
        if d["dep"] == "case":
            tok_purged_pos = [_["pos"] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens][0]
            if tok_purged_pos == "IN":
                return 1
    return 0


def f_character_delta_prune(d):
    character_delta = sum(len(w["word"]) for w in d["source_json"]["tokens"])
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    character_delta -= sum(len(w["word"]) for w in d["source_json"]["tokens"] if w["index"] in cut_tokens)
    return character_delta


def f_len_after_cut_case(d):
    if d["dep"] == "case":
        return f_len_after_cut(d)
    else:
        return 0


def get_delta_helper(d):
    character_delta = sum(len(w["word"]) for w in d["source_json"]["tokens"])
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    character_delta -= sum(len(w["word"]) for w in d["source_json"]["tokens"] if w["index"] in cut_tokens)
    return character_delta


def f_character_delta_prune_case(d):
    if d["dep"] == "case":
        return get_delta_helper(d)
    else:
        return 0


def f_character_delta_prune_nsubj(d):
    if d["dep"] == "nsubj":
        return get_delta_helper(d)
    else:
        return 0

def f_unigram_prune_pos(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)


    if d["op"] == "prune" and len(cut_tokens) == 1:
        pos = [w["pos"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens][0]
        return pos
    return []

def f_unigram_prune_pos_first(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    pos = [w["pos"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens and w["index"] == 1]

    if len(pos) > 0:
        return pos[0]
    else:
        return []

def f_last_word(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)

    if d["op"] == "prune" and len(cut_tokens) == 1:
        word = [w["word"] for w in d["source_json"]["tokens"] if w["index"] not in cut_tokens][0]
        return word
    return ""

def f_is_nmod_and_first_pos_is_prep_and_ends_sentence(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] == "nmod" and f_ends_sentence(d):
        out = [w["pos"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens][0]
        #print out == "IN"
        return out == "IN"
    return False

def f_is_nmod_and_first_pos_is_prep(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] == "nmod":
        out = [w["pos"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens][0]
        #print out == "IN"
        return out == "IN"
    return False

def f_is_nmod_and_ends_sentence(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    if d["dep"] == "nmod" and f_ends_sentence(d):
        return True 
    return False


def f_is_nmod_and_first_word_is_for(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] == "nmod":
        out = [w["word"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens][0]
        return out == "for"
    return False


def f_is_nmod_and_prev_punt_and_next_punct(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    if d["dep"] == "nmod" and f_prev_has_punct(d) and f_next_has_punct(d):
        return True
    return False


def f_next_has_punct_conj(d):
    if d["dep"] == "conj":
        return f_next_has_punct(d)
    else:
        return 0


def f_prev_has_punct_conj(d):
    if d["dep"] == "conj":
        return f_prev_has_punct(d)
    else:
        return 0


def f_prev_has_punct_mark(d):
    if d["dep"] == "mark":
        return f_prev_has_punct(d)
    else:
        return 0

def f_next_has_punct_dep(d):
    if d["dep"] == "dep":
        return f_next_has_punct(d)
    else:
        return 0


def f_prev_has_punct_dep(d):
    if d["dep"] == "dep":
        return f_prev_has_punct(d)
    else:
        return 0


def f_next_has_punct_dobj(d):
    if d["dep"] == "dobj":
        return f_next_has_punct(d)
    else:
        return 0


def f_prev_has_punct_dobj(d):
    if d["dep"] == "dobj":
        return f_prev_has_punct(d)
    else:
        return 0


def f_starts_sentence_nsubj(d):
    if d["dep"] == "nsubj":
        return f_starts_sentence(d)
    else:
        return False


def f_is_nmod_and_next_punct(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    if d["dep"] == "nmod" and f_next_has_punct(d):
        return True
    return False


def f_is_nmod_and_prev_punct(d):
    '''e.g. Bill put his hat on the porch => Bill put his his hat... Feat would fire here '''
    if d["dep"] == "nmod" and f_prev_has_punct(d):
        return True
    return False


def f_len_cut(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return len(cut_tokens)

def f_len_after_cut(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return len([_ for _ in d["source_json"]["tokens"] if _["index"] not in cut_tokens])


def f_tok_after_cut(d):
    cut_tokens = max(h_simulate_op_and_get_ix_affected(d)) + 1
    out = [_ for _ in d["source_json"]["tokens"] if _["index"] == cut_tokens]
    if len(out) > 0:
        return out[0]["lemma"]
    else:
        return "EOS"


def f_tok_after_cut_cv(d):
    cut_tokens = max(h_simulate_op_and_get_ix_affected(d)) + 1
    out = [myself() + _["pos"] for _ in d["source_json"]["tokens"] if _["index"] == cut_tokens]
    if len(out) > 0:
        return [out[0]]
    else:
        return ["EOS"]


def f_len_after_cut_lt_3(d):
    return f_len_after_cut(d) < 3


def f_token_len_is_1_amod(d):
    '''e.g. Welcome  => welcome '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return d["dep"] == 'amod' and len(cut_tokens) == 1


def f_token_len_gt_1(d):
    return len(h_simulate_op_and_get_ix_affected(d)) > 1


def f_token_len_lt_three_and_nmod(d):
    '''e.g. Welcome  => welcome '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] == 'nmod':
        return sum(1 for w in d["source_json"]["tokens"] if w["index"] not in cut_tokens) < 3
    return False


def f_first_cut_word_is_to(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    word_one = [w["word"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens]
    return word_one[0].lower() == "to"


def f_first_cut_word_is_to_xcomp(d):
    if d["dep"] == "xcomp":
        return f_first_cut_word_is_to(d)
    else:
        return False


def f_first_pos_cut(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    word_one = [w["pos"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens][0]
    return word_one


def f_first_cut_is_i(d):
    '''
    useful for nsubj in a few rare cases where workers say yes
    I got ta hunt.
    Got ta hunt.

    '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if len(cut_tokens) == 1 and cut_tokens[0] == 1:
        if [_["word"] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens][0] == "I":
            return True
    return False


def f_first_cut_is_i_nsubj(d):
    '''
    useful for nsubj in a few rare cases where workers say yes
    I got ta hunt.
    Got ta hunt.

    '''
    if d["dep"] == "nsubj":
        return f_first_cut_is_i(d)
    return False


def f_has_appos_aux(d):
    if len(f_unigram_prune_aux_cv(d)) == 1:
        return "'" in f_unigram_prune_cv(d)[0]
    else:
        return False


def f_unigram_prune_aux(d):
    if d["dep"] == "aux":
        return f_unigram_prune_cv(d)
    else:
        return []


def f_aux_has_appos(d):
    '''
    We thought we 'd have an interesting dynamic there.
    We thought we have an interesting dynamic there.
    '''
    if d["dep"] == "aux":
        if len(f_unigram_prune_cv(d)) > 0:
            word = f_unigram_prune_cv(d)[0]
            return "'" in word
    return False


def f_unigram_lemma_prune(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)

    if d["op"] == "prune" and len(cut_tokens) == 1:
        w = [w["lemma"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens][0]
        return w.lower()
    return []


def f_unigram_dep(d):
    '''pos of cut token if unigram prune'''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)

    if d["op"] == "prune" and len(cut_tokens) == 1:
        w = [w["dep"] for w in d["source_json"]["basicDependencies"] if w["dependent"] in cut_tokens][0]
        return w
    return []


def f_has_verb(d):
    pos = [i["pos"] for i in d["source_json"]["tokens"]]
    cut_tokens = h_simulate_op_and_get_ix_affected(d)

    if d["op"] == "prune":
        pos = [w["pos"] for w in d["source_json"]["tokens"] if w["index"] not in cut_tokens]
    return 1 if any([i[0].lower() == "v" for i in pos]) else 0


def f_is_tmod(d):
    return d["dep"] == "nmod:tmod"


def f_len_of_compression(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)

    if d["op"] == "prune":
        return sum(len(w["word"]) for w in d["source_json"]["tokens"] if w["index"] in cut_tokens)
    else:
        raise AssertionError('unexpected thing')


def f_len_of_source(d):
    return sum(len(w["word"]) for w in d["source_json"]["tokens"])


def f_one_tok_cut(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return len(cut_tokens) == 1


def f_tok_len_of_compression_is_one(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)

    if d["op"] == "prune":
        return sum(1 for w in d["source_json"]["tokens"] if w["index"] not in cut_tokens) == 1
    else:
        raise AssertionError('unexpected thing')


def f_min_ix_cut(d):
    '''what is the percentage position of the cut in the sentence: 0 to 1'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    return min(ix_cut)


def f_first_unigram_cut(d):
    ug = [_['word'] for _ in d["source_json"]["tokens"] if _["index"] == f_min_ix_cut(d)]
    return ug[0]


def f_before_first_unigram_cut(d):
    min_cut = f_min_ix_cut(d)
    if min_cut == 1:
        return "SOS"
    return [_['word'] for _ in d["source_json"]["tokens"] if _["index"] == f_min_ix_cut(d) - 1][0]


def f_pos_before_cut_cv_dobj(d):
    if d["dep"] == "dobj":
        min_cut = f_min_ix_cut(d)
        if min_cut == 1:
            return "SOS"
        return [_['pos'] for _ in d["source_json"]["tokens"] if _["index"] == f_min_ix_cut(d) - 1]
    else:
        return []


def f_position(d):
    '''what is the percentage position of the cut in the sentence: 0 to 1'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    min_ = min(ix_cut)
    max_ = max([p["index"] for p in d["source_json"]["tokens"]])
    position = min_/max_
    assert position >=0 and position <= 1
    return position


def f_prev_has_punct_advcl(d):
    if d["dep"] == "advcl":
        return f_prev_has_punct(d)
    else:
        return 0


def f_next_has_punct_advcl(d):
    if d["dep"] == "advcl":
        return f_prev_has_punct(d)
    else:
        return 0


def f_starts_sentence_advcl(d):
    if d["dep"] == "advcl":
        return f_starts_sentence(d)
    else:
        return 0


def f_not_starts_or_ends_sentence_advcl(d):
    if d["dep"] == "advcl":
        return f_not_starts_or_ends_sentence(d)
    else:
        return 0


def f_not_starts_or_ends_sentence_case(d):
    if d["dep"] == "case":
        return f_not_starts_or_ends_sentence(d)
    else:
        return False


def f_next_has_caps(d):
    '''next has caps?'''
    # ix 0 below means char 1
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    next_in_s = [o["word"][0] for o in d["source_json"]["tokens"] if o["index"] == (max(ix_cut) + 1)]
    if len(next_in_s) == 0:
        return False
    if len(next_in_s) == 1:
        return next_in_s[0].isupper()


def f_prev_has_caps(d):
    '''has caps'''
    # ix 0 below means char 1
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    next_in_s = [o["word"][0] for o in d["source_json"]["tokens"] if o["index"] == (min(ix_cut) - 1)]
    if len(next_in_s) == 0:
        return False
    if len(next_in_s) == 1:
        return next_in_s[0].isupper()


def f_prev_has_punct(d):
    '''has caps'''
    # ix 0 below means char 1
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    next_in_s = [o["word"] for o in d["source_json"]["tokens"] if o["index"] == (min(ix_cut) - 1)]
    if len(next_in_s) == 0:
        return False
    if len(next_in_s) == 1:
        return next_in_s[0] in PUNCT


def f_prev_has_punct_xcomp(d):
    if d["dep"] == "xcomp":
        return f_prev_has_punct(d)
    return False


def f_prev_has_punct_case(d):
    if d["dep"] == "case":
        return f_prev_has_punct(d)
    else:
        return False


def f_next_has_punct(d):
    '''has caps'''
    # ix 0 below means char 1
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    next_in_s = [o["word"] for o in d["source_json"]["tokens"] if o["index"] == (max(ix_cut) + 1)]
    if len(next_in_s) == 0:
        return False
    if len(next_in_s) == 1:
        return next_in_s[0] in PUNCT


def f_next_has_punct_case(d):
    if d["dep"] == "case":
        return f_next_has_punct(d)
    else:
        return False


def f_pos_in_cut(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut]
    return w_cut


def f_pos_out_cut(d):
    '''get pos out of cut cut'''

    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [o["pos"] for o in d["source_json"]["tokens"] if o["index"] not in ix_cut]
    return w_cut


def f_lower_case_words_in_cut(d):
    '''get all words in cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)


    w_cut = [o["word"].lower() for o in d["source_json"]["tokens"] if o["index"] in ix_cut]
    return w_cut


def f_is_cc_and_prev_has_punct(d):
    '''

    s: What she has done is prove she's in charge of her sound, her image, and just about everything else. 
    c: What she has done is prove she's in charge of her sound, her image, just about everything else

    '''
    if f_prev_has_punct(d) and d["dep"] == "cc":
        return True
    else:
        return False


def f_prev_has_punct_and_f_next_has_punct(d):
    return f_prev_has_punct(d) and f_next_has_punct(d)


def f_prev_has_punct_or_f_next_has_punct(d):
    return f_prev_has_punct(d) or f_next_has_punct(d)


def f_prev_has_punct_and_f_next_has_punct_conj(d):
    if d["dep"] == "conj":
        return f_prev_has_punct_and_f_next_has_punct(d)
    else:
        return 0


def f_prev_has_punct_or_f_next_has_punct_conj(d):
    if d["dep"] == "conj":
        return f_prev_has_punct_or_f_next_has_punct(d)
    else:
        return 0


def f_is_cc_and_starts_sentence(d):
    '''
    But let me state this as clearly as I can.
    Let me state this as clearly as I can.

    This is a common pattern
    '''
    if d["dep"] == "cc":
        ix_cut = h_simulate_op_and_get_ix_affected(d)
        return ix_cut[0] == 1
    return False


def f_first_word_in_cut(d):
    '''get first word in cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [o["word"].lower() for o in d["source_json"]["tokens"] if o["index"] in ix_cut][0]
    return w_cut


def f_first_word_in_nmod(d):
    if d["dep"] == "nmod":
        return [f_first_word_in_cut(d)]
    else:
        return []


def f_dep_in_cut(d):
    '''get dependencies in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [o["dep"] for o in d["source_json"]["basicDependencies"] if o["dependent"] in ix_cut]
    return w_cut


def f_dep_not_in_cut(d):
    '''get dependencies in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [o["dep"] for o in d["source_json"]["basicDependencies"] if o["dependent"] not in ix_cut]
    return w_cut


def f_level_in_tree(d):
    '''get pos in cut'''
    d, pi, c = bfs(g=d["source_json"], hop_s=0)
    return d[int(d["vertex"])]


def f_dep_out_cut(d):
    '''get pos out of cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [o["dep"] for o in d["source_json"]["basicDependencies"] if o["dependent"] not in ix_cut]
    return w_cut


def f_lemma_in_compression(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    if d["op"] == "prune":
        all_pos = [o["word"] for o in d["source_json"]["tokens"]
                   if o["index"] not in ix_cut]
        return all_pos[0]
    else:
        assert "bad" == "thing"


def f_starts_w_verb_after_cut(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    remaining = [_ for _ in d["source_json"]["tokens"] if _["index"] not in ix_cut][0]
    return remaining['pos'][0].lower() == "v"


def f_starts_w_verb_after_cut_nsubj(d):
    if d["dep"] == "nsubj":
        return f_starts_w_verb_after_cut(d)
    else:
        return 0


def f_prune_starts_w_prp_start_of_sent(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    first_cut = [_ for _ in d["source_json"]["tokens"] if _["index"] in ix_cut and _["index"] == 1]
    if len(first_cut) == 0:
        return False
    return first_cut[0]['pos'] == "PRP"


def f_first_nsubj_off_root(d):

    off_root = [_["dependent"] for _ in d["source_json"]["basicDependencies"] if _['governor'] == 0]
    if len(off_root) == 0: # AH: June 19. If the sentence got heuristic backoff extract then there might not be a root.
        return 0
    off_root = [_["dependent"] for _ in d["source_json"]["basicDependencies"] if _['governor'] == 0][0]
    parent = [_ for _ in d["source_json"]["basicDependencies"] if _["governor"] == off_root and _['dependent'] == int(d["vertex"])]
    parent = [_ for _ in parent if _["dep"] == "nsubj"]
    if d["op"] == "prune" and len(parent) == 1:
        return True
    else:
        return False

def f_next_word_is_said(d):
    '''get pos out of cut'''
    ix_cut = max(h_simulate_op_and_get_ix_affected(d)) + 1
    if d["op"] == "prune":
        try:
            next_word = [o["word"] for o in d["source_json"]["tokens"]
                         if o["index"] == ix_cut][0]
            return next_word.lower() == "said"
        except:
            return 0
    else:
        return 0


def f_nsubj_start_w_I(d):
    return f_starts_sentence(d) and f_unigram_prune_cv(d) == ["I"] and d["dep"] == "nsubj"


def f_r_prune(d):
    '''
    is it an rprune?

    an rprune is a prune of a RHS child. This is a feature of F&A. Is it general?
    '''
    parent = [_["governor"] for _ in d["source_json"]["basicDependencies"] if _["dependent"] == int(d["vertex"])][0]
    sibs = [_["dependent"] for _ in d["source_json"]["basicDependencies"] if _['governor'] == parent if _["dependent"] != int(d["vertex"])]
    if all(_ < int(d["vertex"]) for _ in sibs) and len(sibs) > 0:
        return 1 # yes a RHS prune
    else:
        return 0


def h_get_last_ix_before_cut(d):
    return min(h_simulate_op_and_get_ix_affected(d)) - 1


def get_min_ix_cut(d):
    return min(h_simulate_op_and_get_ix_affected(d))


def get_last_token_before_cut(d):
    ops = [_["word"] for _ in d["source_json"]["tokens"] if _["index"] == h_get_last_ix_before_cut(d)]
    if len(ops) == 1:
        return ops[0]
    else:
        return "SOS"


def f_last_token_before_cut_cv_dobj(d):
    if d["dep"] == "dobj":
        ops = [_["word"] for _ in d["source_json"]["tokens"] if _["index"] == h_get_last_ix_before_cut(d)]
        if len(ops) == 1:
            return ops
        else:
            return ["SOS"]
    else:
        return []


def f_parataxis_and_weird_punct(d):
    if d["dep"] == "parataxis":
        return h_get_word_before_cut(d) in ["--", ":", ";"]
    else:
        return False


def f_dep_and_weird_punct(d):
    if d["dep"] == "dep":
        return h_get_word_before_cut(d) in ["--", ":", ";"]
    else:
        return False


def f_ends_sentence_compoundprt(d):
    if d["dep"] == "compound:prt":
        return f_ends_sentence(d)
    else:
        return 0


def f_ends_sentence_dobj(d):
    if d["dep"] == "dobj":
        return f_ends_sentence(d)
    else:
        return 0


def h_get_word_before_cut(d):
    ops = [_["word"] for _ in d["source_json"]["tokens"] if _["index"] == h_get_last_ix_before_cut(d)]
    if len(ops) == 1:
        return ops[0]
    else:
        return "SOS"


def h_get_pos_before_cut(d):
    ops = [_["pos"] for _ in d["source_json"]["tokens"] if _["index"] == h_get_last_ix_before_cut(d)]
    if len(ops) == 1:
        return ops[0]
    else:
        return "SOS"


def get_min_token_in_cut(d):
    ops = [_["word"] for _ in d["source_json"]["tokens"] if _["index"] == get_min_ix_cut(d)]
    return ops[0]


def h_get_before_and_after(d):
    before = get_last_token_before_cut(d)
    after = get_min_token_in_cut(d)
    return {"before": before, "after": after}


def check_collocation(before_after):
    try:
        with open(COLLOCATION_DIRECTORY + before_after["before"] + '.jsonl', 'r') as inf:
            lns = [_.split("\t") for _ in inf if before_after["after"] in _.split("\t")]
            # lns if of structure ['for', '0.2358447054993353', '2.204971838941199\n']
            # for "planning for"
            lns = [_ for _ in lns if float(_[1]) > 0 and float(_[1]) < RHS_COLLOCATION_THRESHOLD] # i.e. word after ususally comes AFTER
            lns = [_ for _ in lns if float(_[2]) < COLLOCATION_VARIANCE_THRESHOLD]
            return len(lns) > 0
    except IOError:
        if VERBOSE:
            print before_after["before"], "ERROR"
        return False


def f_is_collocation_ish(d):
    before_after = h_get_before_and_after(d)
    return check_collocation(before_after)


def f_is_collocation_ish_neg(d):
    if d["dep"] == "neg":
        return f_is_collocation_ish(d)
    else:
        return 0

def f_is_collocation_ish_acl(d):
    if d["dep"] == "acl":
        return f_is_collocation_ish(d)
    else:
        return 0

def f_pos_before_cut_is_verb(d):
    if h_get_pos_before_cut(d)[0] == "V":
        return True
    return False


def f_pos_before_cut_is_verb_nmod(d):
    if d["dep"] == "nmod":
        return f_pos_before_cut_is_verb(d)
    return False


def f_is_collocation_ish_amod(d):
    if f_is_unigram_amod(d):
        after = h_get_governor_token(d)
        before = f_unigram_prune_cv(d)[0]
        return check_collocation({"before": before, "after": after})
    return False


def f_unigram_prune_cv_case(d):
    if d["dep"] == "case":
        return [myself() +_ for _ in f_unigram_prune_cv(d)]
    else:
        return []


def f_next_has_punct_nmod_poss(d):
    if d["dep"] == 'nmod:poss':
        return f_next_has_punct(d)
    else:
        return 0


def f_unigram_prune_cv_nmod_poss(d):
    if d["dep"] == "nmod_poss":
        return [myself() +_ for _ in f_unigram_prune_cv(d)]
    else:
        return []


def f_unigram_prune_cv_acl(d):
    if d["dep"] == "acl":
        return [myself() +_ for _ in f_unigram_prune_cv(d)]
    else:
        return []


def f_is_collocation_ish_advmod(d):
    if f_is_unigram_advmod(d):
        return f_is_collocation_ish(d)
    return False


def f_is_collocation_ish_nmod(d):
    if d["dep"] == "nmod":
        return f_is_collocation_ish(d)
    else:
        return False


def f_is_collocation_ish_compound(d):
    if d["dep"] == "compound":
        after = h_get_governor_token(d)
        if len(f_unigram_prune_cv(d)) != 1:
            return False
        before = f_unigram_prune_cv(d)[0]
        return check_collocation({"before": before, "after": after})
    else:
        return False


def f_cd_in_out_pos_nummod(d):
    return "CD" in f_pos_out_cut_cv(d)


def get_pos_before_cut_dollar_nummod(d):
    return h_get_pos_before_cut(d)[0] == "$"


def f_next_token_is_percent_nummod(d):
    if d["dep"] == "nummod":
        return f_g_next_token_is_percent(d)
    else:
        return 0


'''
CV features
'''


def f_first_word_in_cut_cv(d):
    return [myself() + get_last_token_before_cut(d)]


def f_pos_in_cut_cv(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut]
    return w_cut


def f_pos_in_cut_cv_dobj(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut]
    return w_cut


def f_pos_in_cut_cv_nsubj(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut]
    return w_cut


def f_nmod_pos_is_their(d):
    if d["dep"] == "nmod:poss":
        return f_unigram_prune_cv(d) == ["their"]
    else:
        return 0


def f_first_pos_in_cut_cv_acl(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut][0:1]
    return w_cut


def f_first_pos_in_cut_cv_nmod_poss(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut][0:1]
    return w_cut


def f_first_pos_in_cut_cv_dobj(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut][0:1]
    return w_cut


def f_pos_out_cut_cv(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] not in ix_cut]
    return w_cut


def f_unigram_prune_cv(d):
    '''
    pos of cut token if unigram prune

    THIS ONE DOES NOT CALL MYSELF() b/c it is is used as part of composition 
    in dep-specific cases
    '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["op"] == "prune" and len(cut_tokens) == 1:
        return [w["word"] for w in d["source_json"]["tokens"] if w["index"] in cut_tokens]
    return []


def f_unigram_prune_cv_generic(d):
    return [myself() + _ for _ in f_unigram_prune_cv(d)]


def f_unigram_prune_cv_dobj(d):
    if d["dep"] == "dobj":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    return []


def f_unigram_prune_cv_neg(d):
    '''
    pos of cut token if unigram prune

    THIS ONE DOES NOT CALL MYSELF() b/c it is is used as part of composition 
    in dep-specific cases
    '''
    if d["dep"] == "neg":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    return []


def f_unigram_prune_cv_mark(d):
    '''pos of cut token if unigram prune'''
    if d["dep"] == "mark":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    return []


def f_unigram_prune_advmod_cv(d):
    '''pos of cut token if unigram prune'''
    if d["dep"] == "advmod":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    return []


def f_unigram_prune_aux_cv(d):
    if d["dep"] == "aux":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    else:
        return []


def f_ending_3_cv(d):
    '''last three letters of a single op, useful for amod :) !! '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if len(cut_tokens) == 1:
        return [myself() + _["word"][-3:] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens]
    else:
        return [""]


def f_ending_3_amod_cv(d):
    '''last three letters of a single op, useful for amod :) !! '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if d["dep"] != "amod":
        return []
    if len(cut_tokens) == 1:
        return [myself() + _["word"][-3:] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens]
    else:
        return []


def f_ending_2_cv(d):
    '''last three letters of a single op, useful for amod :) !! '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if len(cut_tokens) == 1:
        return [myself() + _["word"][-2:] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens]
    else:
        return [""]


def f_previous_pos_cv(d):
    ''' '''
    return [myself() + h_get_pos_before_cut(d)]


def f_previous_pos_cv_case(d):
    ''' '''
    if d["dep"] == "case":
        return [myself() + _ for _ in h_get_pos_before_cut(d)]
    else:
        return []


def f_unigram_prune_cv_nsubj(d):
    if d["dep"] == "nsubj":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    else:
        return []


def f_unigram_prune_cv_cop(d):
    if d["dep"] == "cop":
        return [myself() + _ for _ in f_unigram_prune_cv(d)]
    else:
        return []


def f_pos_tag_of_determiner_parent_cv(d):
    if d["dep"] == "det":
        gov = [int(_["governor"]) for _ in
               d["source_json"]["basicDependencies"]
               if _["dependent"] == int(d["vertex"])][0]
        pos = [myself() + _["pos"] for _ in d["source_json"]["tokens"] if int(_["index"]) == gov]
        return pos
    else:
        return []


def f_unigram_prune_is_dash_parens(d):
    return len(set(f_unigram_prune_cv(d)) & set(["-LRB-", "-RRB-", "-LSB-", "-RSB-"])) > 0

'''
Determiner features
'''

def f_pos_tag_of_determiner_parent_nns(d):
    if d["dep"] == "det":
        gov = [int(_["governor"]) for _ in 
               d["source_json"]["basicDependencies"] 
               if _["dependent"] == int(d["vertex"])][0]
        pos = [_["pos"] for _ in d["source_json"]["tokens"] if int(_["index"]) == gov]
        return pos in ["NNS", "NNPS"]
    else:
        return False


def f_dep_is_det_and_next_token_is_plural_noun(jdoc):
    '''
    op is determiner and next token is NNS or NNPS

    mean yes rate for when this feature is True is 64% and mean yes rate
    when not true is 30%
    '''
    if jdoc["dep"] == "det":
        ix = h_simulate_op_and_get_ix_affected(jdoc)[0]
        answ = [i["pos"] in ["NNS", "NNPS"] for i in jdoc["source_json"]["tokens"] if i["index"] == ix + 1][0]
        return answ
    else:
        return False

#########################################################################
#
# Global, non-interaction features go here.
#
# each has the tag f_g to start the function
#
#########################################################################


def f_g_starts_sentence(d):
    return f_starts_sentence(d)


def f_g_ends_sentence(d):
    return f_ends_sentence(d)


def f_g_not_starts_or_ends_sentence(d):
    return not (f_starts_sentence(d) or f_ends_sentence(d))


def f_g_prev_has_punct(d):
    return f_prev_has_punct(d)


def f_g_next_has_punct(d):
    return f_next_has_punct(d)


def f_g_prev_has_punct_and_f_next_has_punct(d):
    return f_next_has_punct(d) and f_prev_has_punct(d)


def f_g_is_collocation_ish(d):
    colloc = f_is_collocation_ish(d)
    if "_unit_id" in d.keys():
        logger.info("{},{},{}".format(d["dep"], d["_unit_id"], colloc))
 
    if f_is_collocation_ish(d):
        if "_unit_id" in d.keys():
            logger.info(d['source'] + "+" + d["compression"] + "|examples")
    return f_is_collocation_ish(d)


def f_g_f_character_delta_prune(d):
    return f_character_delta_prune(d)


def f_g_len_after_cut(d):
    return f_len_after_cut(d)


def f_g_governor_has_appos(d):
    return "'" in h_get_governor_token(d)


def f_g_next_token_is_percent(d):
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    next_tok = max(ix_cut) + 1
    remaining = [_ for _ in d["source_json"]["tokens"] if _["index"] == next_tok]
    if len(remaining) == 0:
        return 0
    else:
        return remaining[0]["word"] == "percent"


def f_g_starts_w_verb_after_cut(d):
    return f_starts_w_verb_after_cut(d)


def f_g_is_short(d):
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    return len(cut_tokens) < 5


def f_g_is_their(d):
    return f_unigram_prune_cv(d) == ["their"]


def f_g_cd_in_out_pos(d):
    return "CD" in f_pos_out_cut_cv(d)


def f_g_first_cut_word_is_to(d):
    '''pos of cut token if unigram prune'''
    return f_first_cut_word_is_to(d)


def f_g_clark_lapin_slor_of_c(d):
    return f_slor_of_c(d)


# Global CV features here


def f_g_worker_id_cv(d):
    return [myself() + str(d["_worker_id"])]


def f_g_dep_cv(d):
    '''pos of cut token if unigram prune'''
    return [myself() + d["dep"]]


def f_g_pos_in_cut_cv(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut]
    return w_cut


def f_g_first_pos_in_cut_cv(d):
    '''get pos in the cut'''
    ix_cut = h_simulate_op_and_get_ix_affected(d)
    w_cut = [myself() + o["pos"] for o in d["source_json"]["tokens"] if o["index"] in ix_cut][0:1]
    return w_cut


def f_g_ending_3_cv(d):
    '''last three letters of a single op, useful for amod :) !! '''
    cut_tokens = h_simulate_op_and_get_ix_affected(d)
    if len(cut_tokens) == 1:
        return [myself() + _["word"][-3:] for _ in d["source_json"]["tokens"] if _["index"] in cut_tokens]
    else:
        return [""]


def f_g_pos_tag_of_parent_cv(d):
    gov = [int(_["governor"]) for _ in
           d["source_json"]["basicDependencies"]
           if _["dependent"] == int(d["vertex"])][0]
    pos = [myself() + _["pos"] for _ in d["source_json"]["tokens"] 
           if int(_["index"]) == gov]
    return pos


def f_g_previous_pos_cv(d):
    ''' '''
    return [myself() + h_get_pos_before_cut(d)]


def f_g_unigram_prune_cv(d):
    ''' '''
    return [myself() + i for i in f_unigram_prune_cv(d)]


### multi op features ###

from pylru import lrudecorator


@lrudecorator(100)
def get_cache_and_predictor():
    from round2.modeling_utilities import read_from_cache
    CACHE = read_from_cache()
    PREDICTOR = FigureEightPredictor(cache=CACHE)
    return CACHE, PREDICTOR


def f_g_m_n_score(d):
    assert "multiop" in d["batchno"]
    v = json.loads(d["vertex"])
    assert type(v) == list
    CACHE, PREDICTOR = get_cache_and_predictor()
    bv = BVCompressor(predictor=PREDICTOR,
                      r=None,
                      Q=[],
                      sentence=None,
                      verbose=False
                      )

    return bv.readability(source=copy.deepcopy(d["source_json"]),
                          vertexes=v,
                          predictor=PREDICTOR)


def f_g_m_n_ops(d):
    assert "multiop" in d["batchno"]
    v = json.loads(d["vertex"])
    return -1 * len(v)


def f_g_m_n_r(d):
    assert "multiop" in d["batchno"]
    return int(d["r"])


def f_g_m_norm_lp_multiop(d):
    assert "multiop" in d["batchno"]
    s = d['source_json']
    cut_tokens = h_simulate_op_and_get_ix_affected_multi_op(d)
    tokens = ["SOS"] + [o["word"] for o in s["tokens"] if o["index"] not in cut_tokens] + ["EOS"]
    return h_norm_lp(tokens)


def f_g_m_norm_lp_multiop_binary(d):
    assert "multiop" in d["batchno"]
    s = d["source_json"]
    cut_tokens = h_simulate_op_and_get_ix_affected_multi_op(d)
    sw = ["SOS"] + [o["word"] for o in s["tokens"]] + ["EOS"]
    cw = ["SOS"] + [o["word"] for o in s["tokens"] if o["index"] not in cut_tokens] + ["EOS"]
    return (h_norm_lp(cw) - h_norm_lp(sw)) > 0


def f_g_m_min_op(d):
    CACHE, PREDICTOR = get_cache_and_predictor()

    assert "multiop" in d["batchno"]

    s = d['source_json']
    gs = BVCompressor(predictor=PREDICTOR,
                      r=100,
                      Q=[],
                      sentence=copy.deepcopy(s),
                      verbose=False
                      )

    dv = d['vertex'].decode('string-escape').strip('"')

    worst_op = gs.worst_op(source=copy.deepcopy(s),
                           predictor=PREDICTOR,
                           vertexes=json.loads(dv)
                           )

    return worst_op


def f_g_m_dep_cv(d):
    '''pos of cut token if unigram prune'''
    dv = d['vertex'].decode('string-escape').strip('"')
    dv = [int(_) for _ in json.loads(dv)]
    out = [myself() + _["dep"] for _ in d["source_json"]["basicDependencies"] if int(_["governor"]) in dv]
    return out


@lrudecorator(10)
def get_cola_preds():
    with open("cola/cola.preds.single.jsonl", "r") as inf:
        single = json.load(inf)

    with open("cola/cola.preds.multi.jsonl", "r") as inf:
        multi_op = json.load(inf) 
    return single, multi_op


def f_g_cola_c_multi(d):
    unit_id = int(d["_unit_id"])
    single, multi_op = get_cola_preds()
    c_score = multi_op[str(unit_id) + "compression"]["score"]
    s_score = multi_op[str(unit_id) + "source"]["score"]
    return float(c_score) - float(s_score)


def f_g_real_fake_multi(d):
    '''
    predicted positive from real/fake detector in the real/fake repo
    real_fake/all_preds.json needs to get filled
    '''
    with open("real_fake/all_preds.json", "r") as inf:
        dt = json.load(inf)
    unit_id = d["_unit_id"]
    return float(dt[unit_id])


def f_g_cola_c_single_op_delta(d):
    unit_id = int(d["_unit_id"])
    single_op, multi_op = get_cola_preds()
    c_score = single_op[str(unit_id) + "compression"]["score"]
    s_score = single_op[str(unit_id) + "source"]["score"]
    return float(c_score) - float(s_score)


def f_g_cola_c_single_op_delta_log(d):
    unit_id = int(d["_unit_id"])
    single_op, multi_op = get_cola_preds()
    c_score = single_op[str(unit_id) + "compression"]["score"]
    s_score = single_op[str(unit_id) + "source"]["score"]
    return math.log(float(c_score)) - math.log(float(s_score))


def f_g_cola_c_single_op_indicator(d):
    unit_id = int(d["_unit_id"])
    single_op, multi_op = get_cola_preds()
    c_score = single_op[str(unit_id) + "compression"]["score"]
    s_score = single_op[str(unit_id) + "source"]["score"]
    return float(c_score) > float(s_score)


def f_g_cola_c_single_op(d):
    unit_id = int(d["_unit_id"])
    single_op, multi_op = get_cola_preds()
    c_score = single_op[str(unit_id) + "compression"]
    return float(c_score["score"])


def f_g_cola_c_single_op_log(d):
    unit_id = int(d["_unit_id"])
    single_op, multi_op = get_cola_preds()
    c_score = single_op[str(unit_id) + "compression"]
    return math.log(float(c_score["score"]))

