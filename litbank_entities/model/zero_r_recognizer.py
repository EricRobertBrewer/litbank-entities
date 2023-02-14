from typing import List

from litbank_entities import litbank
from litbank_entities.model import recognizer


class ZeroREntityRecognizer(recognizer.EntityRecognizer):
    """
    Tagger which guesses that every sentence is an entire entity for each category.
    """

    def train(self, sentence_tokens: List[List[str]], sentence_labels: List[List[List[str]]]):
        pass

    def predict(self, sentence_tokens: List[List[str]]) -> List[List[List[str]]]:
        return [[litbank.ENTITY_TAGS[:6]] +
                [litbank.ENTITY_TAGS[6:12] for _ in range(len(tokens)-1)]
                for tokens in sentence_tokens]
