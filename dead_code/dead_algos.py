
class NeuralNetworkTransitionGreedyPlusLength:
    def __init__(self, archive_loc, model_name, predictor_name, T=.5, alpha=.5, query_focused=True):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.archive = archive
        self.T = T
        self.query_focused = query_focused
        self.alpha = float(alpha)
        self.predictor = Predictor.from_archive(archive, predictor_name)

    def predict_proba(self, original_s, vertex, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        provisional_label = "p"
        toks = get_encoded_tokens(provisional_label, state,
                                  original_s, vertex)

        txt = " ".join([_["word"] for _ in toks])

        instance = self.predictor._dataset_reader.text_to_instance(txt, True,
                                                                   "1")
        would_be_pruned = len(" ".join([_["word"] for _ in state["tokens"]
                              if _["index"] in
                              dfs(state, hop_s=vertex, D=[])]))

        total_pruned = len(" ".join([_["word"] for _ in state["tokens"]]))

        pct_pruned = would_be_pruned/total_pruned

        pred_labels = self.archive.model.vocab.get_index_to_token_vocabulary("labels")
        op2n = {v:k for k,v in pred_labels.items()}
        pred = self.predictor.predict_instance(instance)
        return np.mean([self.alpha * pred["class_probabilities"][op2n["1"]],
                       (1 - self.alpha) * pct_pruned])

    def predict_vertexes(self, jdoc, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        assert self.query_focused
        return {_["index"]: self.predict_proba(original_s=jdoc,
                                               vertex=_["index"],
                                               state=state)
                for _ in state["tokens"] if not prune_deletes_q(_["index"],
                                                                jdoc)}

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def heuristic_extract(self, jdoc):
        '''
        return the lowest vertex in the tree that contains the query terms
        '''
        from_root = [_['dependent'] for _ in jdoc["basicDependencies"] if _['governor'] == 0][0]
        best = from_root
        def tok_is_verb(vertex):
            gov = [o["pos"][0] for o in jdoc["tokens"] if o["index"] == v][0]
            return gov[0].lower() == "v"
        for v in get_walk_from_root(jdoc):  # bfs
            children = dfs(g=jdoc, hop_s=v, D=[])
            # the verb heuristic is b/c the min governing tree is often just Q itself
            if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
                best = v
        return best

    def init_state(self, jdoc):
        '''init to the governing subtree'''
        topv = self.heuristic_extract(jdoc)
        if jdoc["q"] != []:
            short_tree = dfs(g=jdoc, hop_s=topv, D=[])
            toks_to_start = [i for i in jdoc["tokens"] if i["index"] in short_tree]
            deps_to_start = [i for i in jdoc["basicDependencies"] if
                             i["dependent"] in short_tree
                             and i["governor"] in short_tree]
            state = {"tokens": toks_to_start, "basicDependencies": deps_to_start}
        else:
            state = {"tokens": jdoc["tokens"], "basicDependencies": jdoc["basicDependencies"]}
        return state

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = 0

        state = self.init_state(jdoc)
        prunes = 0
        while length != prev_length and length > int(jdoc["r"]):
            prunes += 1
            vertexes = list(self.predict_vertexes(jdoc=jdoc, state=state).items())
            nops += len(vertexes)
            vertexes.sort(key=lambda x: x[1], reverse=True)
            if len(vertexes) == 0:
                print("huh")
                break
            vertex, prob = vertexes[0]
            prune(g=state, v=vertex)
            prev_length = length
            length = self.get_char_length(state)
        length = self.get_char_length(state)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in state["tokens"]]
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "nops": nops,
                    "prunes": prunes
                    }
        else:
            return {"y_pred": "could not find a compression",
                    "nops": nops
                    }


class WorstCaseCompressor:
    '''
    Performs the absolutely worst number of ops

    Prunes only a singleton vertex
    '''
    def __init__(self):
        pass

    def predict_len(self, original_s, vertex, state):
        '''
        the length is just the number of children
        '''
        children = [o for o in state["basicDependencies"]
                    if o["governor"] == vertex]
        return len(children)

    def predict_vertexes(self, jdoc, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        return {_["index"]: self.predict_len(original_s=jdoc,
                                             vertex=_["index"],
                                             state=state)
                for _ in state["tokens"] if not prune_deletes_q(_["index"],
                                                                jdoc)}

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def heuristic_extract(self, jdoc):
        '''
        return the lowest vertex in the tree that contains the query terms
        '''
        from_root = [_['dependent'] for _ in jdoc["basicDependencies"] if _['governor'] == 0][0]
        best = from_root
        def tok_is_verb(vertex):
            gov = [o["pos"][0] for o in jdoc["tokens"] if o["index"] == v][0]
            return gov[0].lower() == "v"
        for v in get_walk_from_root(jdoc):  # bfs
            children = dfs(g=jdoc, hop_s=v, D=[])
            # the verb heuristic is b/c the min governing tree is often just Q itself
            if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
                best = v
        return best

    def init_state(self, jdoc):
        '''init to the governing subtree'''
        topv = self.heuristic_extract(jdoc)
        if jdoc["q"] != []:
            short_tree = dfs(g=jdoc, hop_s=topv, D=[])
            toks_to_start = [i for i in jdoc["tokens"] if i["index"] in short_tree]
            deps_to_start = [i for i in jdoc["basicDependencies"] if
                             i["dependent"] in short_tree
                             and i["governor"] in short_tree]
            state = {"tokens": toks_to_start, "basicDependencies": deps_to_start}
        else:
            state = {"tokens": jdoc["tokens"], "basicDependencies": jdoc["basicDependencies"]}

        # this line is needed for the "For roughly two-thirds of sentences ..."
        # logger.info("V init sentence ", len(jdoc["tokens"]) == len(state["tokens"]))

        # this line is needed for the "In more than 95% of sentences, the compression system..."
        #r = " ".join([o["word"] for o in state["tokens"]])
        #logger.info("init > r {}".format(len(r) > int(jdoc["r"])))
        return state

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = []

        state = self.init_state(jdoc)
        prunes = 0
        while length != prev_length and length > int(jdoc["r"]):
            prunes += 1
            vertexes = list(self.predict_vertexes(jdoc=jdoc, state=state).items())
            nops.append(len(vertexes))
            vertexes.sort(key=lambda x: x[1])
            if len(vertexes) == 0:
                print("huh")
                break
            vertex, length = vertexes[0]
            assert length == 0
            prune(g=state, v=vertex)
            prev_length = length
            length = self.get_char_length(state)
        length = self.get_char_length(state)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in state["tokens"]]
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "nops": nops,
                    "prunes": prunes
                    }
        else:
            return {"y_pred": "could not find a compression",
                    "nops": nops
                    }



### this one does not work well
class NeuralNetworkPredictThenPrune:
    def __init__(self, archive_loc, model_name, predictor_name, query_focused=True):
        assert type(archive_loc) == str
        archive = load_archive(archive_file=archive_loc)
        self.archive = archive
        self.query_focused = query_focused
        self.predictor = Predictor.from_archive(archive, predictor_name)

    def predict_proba(self, original_s, vertex, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        provisional_label = "p"
        toks = get_encoded_tokens(provisional_label, state,
                                  original_s, vertex)

        txt = " ".join([_["word"] for _ in toks])

        instance = self.predictor._dataset_reader.text_to_instance(txt, True,
                                                                   "1")

        pred_labels = self.archive.model.vocab.get_index_to_token_vocabulary("labels")
        op2n = {v:k for k,v in pred_labels.items()}
        pred = self.predictor.predict_instance(instance)
        return pred["class_probabilities"][op2n["1"]]

    def predict_vertexes(self, jdoc, state):
        '''
        what is probability that this vertex is prunable,
        according to transition-based nn model
        '''
        assert self.query_focused
        return {_["index"]: self.predict_proba(original_s=jdoc,
                                               vertex=_["index"],
                                               state=state)
                for _ in state["tokens"] if not prune_deletes_q(_["index"],
                                                                jdoc)}

    def get_char_length(self, jdoc):
        assert type(jdoc["tokens"][0]["word"]) == str
        return len(" ".join([_["word"] for _ in jdoc["tokens"]]))

    def heuristic_extract(self, jdoc):
        '''
        return the lowest vertex in the tree that contains the query terms
        '''
        from_root = [_['dependent'] for _ in jdoc["basicDependencies"] if _['governor'] == 0][0]
        best = from_root
        def tok_is_verb(vertex):
            gov = [o["pos"][0] for o in jdoc["tokens"] if o["index"] == v][0]
            return gov[0].lower() == "v"
        for v in get_walk_from_root(jdoc):  # bfs
            children = dfs(g=jdoc, hop_s=v, D=[])
            if all(i in children for i in jdoc["q"]) and tok_is_verb(v):
                best = v
        return best

    def predict(self, jdoc):
        '''
        return a compression that preserves q and respects r
        '''
        prev_length = 0
        length = self.get_char_length(jdoc)
        orig_toks = jdoc["original_ix"]
        nops = 0
        topv = self.heuristic_extract(jdoc)

        if jdoc["q"] != []:
            short_tree = dfs(g=jdoc, hop_s=topv, D=[])
            toks_to_start = [i for i in jdoc["tokens"] if i["index"] in short_tree]
            deps_to_start = [i for i in jdoc["basicDependencies"] if
                             i["dependent"] in short_tree
                             and i["governor"] in short_tree]
            state = {"tokens": toks_to_start, "basicDependencies": deps_to_start}
        else:
            state = {"tokens": jdoc["tokens"], "basicDependencies": jdoc["basicDependencies"]}
        vertexes = list(self.predict_vertexes(jdoc=jdoc, state=state).items())
        nops += len(vertexes)
        while length != prev_length and length > int(jdoc["r"]):
            vertexes.sort(key=lambda x: x[1], reverse=True)
            vertex, prob = vertexes[0]
            prune(g=state, v=vertex)
            prev_length = length
            length = self.get_char_length(state)
            vertexes.remove((vertex,prob))
        length = self.get_char_length(state)
        if length <= int(jdoc["r"]):
            remaining_toks = [_["index"] for _ in state["tokens"]]
            return {"y_pred": [_ in remaining_toks for _ in orig_toks],
                    "nops": nops
                    }
        else:
            return {"y_pred": "could not find a compression",
                    "nops": nops
                    }

