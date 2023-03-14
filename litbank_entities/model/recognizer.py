from typing import List


class EntityRecognizer:

    def __init__(self, categories):
        self.categories = categories

    def train(self, sentence_tokens: List[List[str]], sentence_labels: List[List[List[str]]]):
        """
        Model the relationship between the tokens in each sequence (sentence) and
        the corresponding labels.

        :param sentence_tokens: Sequences of tokens.
        :param sentence_labels: BIO (Beginning, Inside, Other) tags sub-typed with
        one of six entity categories. Because tagged phrases can be nested or
        overlap, each label is multidimensional.
        """
        raise NotImplementedError()

    def predict(self, sentence_tokens: List[List[str]]) -> List[List[List[str]]]:
        """
        Determine the corresponding BIO tags for the given sequences.

        :param sentence_tokens: Sequences of tokens.
        :return: List of multidimensional labels.
        """
        raise NotImplementedError()

    def save_model(self, dir_):
        raise NotImplementedError()

    def load_model(self, dir_):
        raise NotImplementedError()
