# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
#FiveSegReader
from nn.models import TwoEmbedsClassifier


class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = TwoEmbedsClassifier()
        instances = ensure_list(reader.read('tests/fixtures/validation_3way.jsonl'))
