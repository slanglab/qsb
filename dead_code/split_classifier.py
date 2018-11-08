from typing import Dict, Optional
import numpy
from overrides import overrides
import json
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
import numpy as np
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("split_classifier")
class AcademicPaperClassifier(Model):
    """
    This ``Model`` performs text classification for an academic paper.  We assume we're given a
    title and an abstract, and we predict some output label.

    The basic model structure: we'll embed the title and the abstract, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    title_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    abstract_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder, 
                 abstract_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AcademicPaperClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.labelv = self.vocab.get_index_to_token_vocabulary("labels")
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != abstract_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }

        # {"0": "NA", "1": "p"} self.labelv

        '''
        n_samples = 2519659
        n_classes = 2
        '''

        # http://scikit-learn.org/dev/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
        # The balanced mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data: n_samples / (n_classes * np.bincount(y))

        import socket
        if socket.gethostname() == "hobbes":
            DIR = "preproc/"
        else:
            DIR = "/mnt/nfs/work1/brenocon/ahandler/qsr/"

        with open(DIR + "full_counts.txt", "r") as inf:
            dt4 = inf.read().split("\n")
            dt4 = [_.strip().split() for _ in dt4 if len(_) > 0]
            dt4 = {v.strip('"'): int(k) for k, v in dt4}

        with open(DIR + "/2way_counts.txt", "r") as inf:
            dt2 = inf.read().split("\n")
            dt2 = [_.strip().split() for _ in dt2 if len(_) > 0]
            dt2 = {v.strip('"'): int(k) for k, v in dt2}

        n_classes = len(self.labelv)
        if n_classes == 2:
            dt = dt2
        elif n_classes == 4:
            dt = dt4
        else:
            assert "bad" == "thing"

        n_samples = sum(dt.values())
        a = np.zeros(n_classes, dtype=np.float32)
        print(self.labelv)
        for i in range(n_classes):
            a[i] = n_samples / (dt[self.labelv[i]] * n_classes)
        with open("wut", "w") as of:
            of.write(np.array_str(a)) 
        a = torch.from_numpy(a)
        self.loss = torch.nn.CrossEntropyLoss(weight=a)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        embedded_abstract = self.text_field_embedder(sentence)
        abstract_mask = util.get_text_field_mask(sentence)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)

        values, indices = torch.max(encoded_abstract, 1)

        logits2 = self.classifier_feedforward(values)

        output_dict = {'logits': logits2}
        if label is not None:
            loss = self.loss(logits2, label)
            for metric in self.metrics.values():
                metric(logits2, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = output_dict['logits']
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
