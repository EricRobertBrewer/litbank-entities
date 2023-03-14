from typing import List

from litbank_entities import litbank
from litbank_entities.model import recognizer


class ZeroREntityRecognizer(recognizer.EntityRecognizer):
    """
    Tagger which guesses that every sentence is an entire entity for each category.
    """

    def __init__(self, categories):
        super().__init__(categories)

    def train(self, sentence_tokens: List[List[str]], sentence_labels: List[List[List[str]]]):
        pass

    def predict(self, sentence_tokens: List[List[str]]) -> List[List[List[str]]]:
        return [[['B-{}'.format(category) for category in self.categories]] +
                [['I-{}'.format(category) for category in self.categories] for _ in range(len(tokens)-1)]
                for tokens in sentence_tokens]

    def save_model(self, dir_):
        pass

    def load_model(self, dir_):
        pass
