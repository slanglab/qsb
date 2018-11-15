from typing import Dict
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("binary_reader")
class SemanticScholarDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the label of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                sentence = " ".join([_["word"] for _ in
                                    paper_json["tokens"]])

                # reduce all labels to 0 or 1, sort of if prune or
                # extract downstream. This is to avoid multiple
                # vocabuliaries, np/p vs ne/e
                if paper_json["label"] not in ["p", "e"]:
                    label = "0"
                else:
                    label = "1"
                if "p" in paper_json["label"]:
                    is_prune = True
                else:
                    is_prune = False
                yield self.text_to_instance(sentence, is_prune, label)

    @overrides
    def text_to_instance(self, sentence: str, is_prune: bool, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        fields = {'sentence': sentence_field}
        fields["is_prune"] = MetadataField(is_prune)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


@DatasetReader.register("markup_5_segment_reader")
class FiveSegReader(DatasetReader):
    """
    - Read in the fields of the markup and break it up into 5 fields
      This is to allow different embeddings matrixes for each of the fields,
      especially the special tags

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                sentence = " ".join([_["word"] for _ in
                                    paper_json["tokens"]])

                # reduce all labels to 0 or 1, sort of if prune or
                # extract downstream. This is to avoid multiple
                # vocabuliaries, np/p vs ne/e
                if paper_json["label"] not in ["p", "e"]:
                    label = "0"
                else:
                    label = "1"
                if "p" in paper_json["label"]:
                    is_prune = True
                else:
                    is_prune = False

                before, after = [ino for ino,i in 
                                 enumerate(paper_json["tokens"]) if "OOV" in i["word"] 
                                 and "proposed" in i["word"]]

                toks = paper_json["tokens"]
                toks.sort(key=lambda x:x["index"])

                def stringify(seq):
                    return " ".join([_["word"] for _ in toks[0:before]])

                vl = stringify(toks[0:before])
                vr = stringify(toks[after+1:])
                tv = stringify(toks[before + 1:after])
                bracket1 = stringify([toks[before]])
                bracket2 = stringify([toks[after]])

                yield self.text_to_instance(vl,vr,tv,bracket1,bracket2,is_prune, label)

    @overrides
    def text_to_instance(self, vl: str,
                         vr: str,
                         tv: str,
                         bracket2: str,
                         bracket1: str,
                         is_prune: bool,
                         label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_vl = self._tokenizer.tokenize(vl)
        tokenized_vr = self._tokenizer.tokenize(vr)
        tokenized_tv = self._tokenizer.tokenize(tv)
        tokenized_bracket2 = self._tokenizer.tokenize(bracket2)
        tokenized_bracket1 = self._tokenizer.tokenize(bracket1)

        vl_field = TextField(tokenized_vl, self._token_indexers)
        vr_field = TextField(tokenized_vr, self._token_indexers)
        tv_field = TextField(tokenized_tv, self._token_indexers)
        bracket1_field = TextField(tokenized_bracket1, self._token_indexers)
        bracket2_field = TextField(tokenized_bracket2, self._token_indexers)

        fields = {'v1': vl_field,
                  'v2': vr_field,
                  'tv': tv_field,
                  'b1': bracket1_field,
                  'b2': bracket2_field
                  }
        fields["is_prune"] = MetadataField(is_prune)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
