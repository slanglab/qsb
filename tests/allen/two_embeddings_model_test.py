# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class TwoEmbedsClassifier(ModelTestCase):
    def setUp(self):
        super(TwoEmbedsClassifier, self).setUp()
        self.set_up_model('tests/fixtures/two_embed.json',
                          'tests/fixtures/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)