class FA2013Compressor:

    '''
    This implements a query query_focused compression w/ F and A
    '''

    def __init__(self, weights):
        from ilp2013.fillipova_altun_supporting_code import get_all_vocabs
        self.weights = weights
        self.vocab = get_all_vocabs()

    def predict(self, original_s):
        '''
        run the ILP
        '''

        r = int(original_s["r"])

        original_indexes = [_["index"] for _ in original_s["tokens"]]

        transformed_s = filippova_tree_transform(copy.deepcopy(original_s))

        output = run_model(transformed_s,
                           vocab=self.vocab,
                           weights=self.weights,
                           Q=original_s["q"],
                           r=r)

        # note: not necessarily a unique set b/c multi edges possible to same
        # vertex after the transform. Set() should not affect eval, if you look
        # at the code in get_pred_y
        predicted_compression = set([o['dependent'] for o in output["get_Xs"]])
        y_pred = get_pred_y(predicted_compression=predicted_compression,
                            original_indexes=original_indexes)

        assert all([i in predicted_compression for i in original_s["q"]])
        assert len(output["compressed"]) <= original_s["r"]
        return {"y_pred": y_pred,
                "compression": output["compressed"],
                "nops": -19999999  # whut to do here????
                }


class FA2013CompressorStandard:

    '''
    This implements a standard version of F and A
    '''

    def __init__(self, weights):
        from ilp2013.fillipova_altun_supporting_code import get_all_vocabs
        self.weights = weights
        self.vocab = get_all_vocabs()

    def predict(self, original_s):
        '''
        run the ILP
        '''

        original_indexes = [_["index"] for _ in original_s["tokens"]]

        transformed_s = filippova_tree_transform(copy.deepcopy(original_s))

        # "To get comparable
        # results, the unsupervised and our systems used
        # the same compression rate: for both, the requested
        # maximum length was set to the length of the headline."
        r = len(original_s["headline"])

        output = run_model(transformed_s,
                           vocab=self.vocab,
                           weights=self.weights,
                           r=r)

        predicted_compression = [o['dependent'] for o in output["get_Xs"]]
        y_pred = get_pred_y(predicted_compression=predicted_compression,
                            original_indexes=original_indexes)

        return {"y_pred": y_pred,
                "compression": output["compressed"],
                "nops": -19999999  # whut to do here????
                }
