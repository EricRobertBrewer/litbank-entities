import operator
from typing import List

import numpy as np

from litbank_entities import linguistics, litbank
from litbank_entities.model import recognizer


class HMMEntityRecognizer(recognizer.EntityRecognizer):
    """
    Hidden Markov Model implementation.

    From `https://web.stanford.edu/~jurafsky/slp3/8.pdf`.
    """

    def __init__(self, categories, ignore_nested=False):
        if categories != litbank.ENTITY_CATEGORIES:
            raise ValueError('HMM needs to be refactored to allow category selection. For now, select all categories.')
        super().__init__(categories)
        self.ignore_nested = ignore_nested

        self.v = None  # Vocabulary and frequencies.
        self.n = None  # Size of vocabulary, |V|.

        self.A = None  # Transition probabilities.
        self.p = None  # Initial state probabilities.
        self.B = None  # Emission probabilities.

    def train(self, sentence_tokens: List[List[str]], sentence_labels: List[List[List[str]]]):
        self.v = linguistics.get_vocabulary_counts(sentence_tokens)
        self.n, X = linguistics.get_n_sentence_token_ids(sentence_tokens, self.v)
        Y = [[[litbank.ENTITY_LABEL_TO_ID[nest_label]
               for nest_label in label]
              for label in labels]
             for labels in sentence_labels]

        Q = litbank.ENTITY_LABELS
        self.A = np.zeros((len(Q), len(Q)))
        self.p = np.zeros(len(Q))
        self.B = np.zeros((len(Q), self.n))

        if self.ignore_nested:
            self._train_counts_columnar(X, Y)
        else:
            self._train_counts_nested(X, Y)
        self.p /= sum(self.p)
        for q in range(len(self.A)):
            self.A[q] /= sum(self.A[q])
            self.B[q] /= sum(self.B[q])

    def _train_counts_columnar(self, X, Y):
        # Drop nested labels.
        Y = [[label[0] for label in y] for y in Y]
        for s, y in enumerate(Y):
            x = X[s]
            self.p[y[0]] += 1
            self.B[y[0], x[0]] += 1
            for t in range(1, len(y)):
                self.A[y[t-1], y[t]] += 1
                self.B[y[t], x[t]] += 1

    def _train_counts_nested(self, X, Y):
        q_out = litbank.ENTITY_LABEL_TO_ID['O']
        for s, y in enumerate(Y):
            x = X[s]
            # Note each unique initial state.
            for q in set(y[0]):
                self.p[q] += 1
                self.B[q, x[0]] += 1
            for t in range(1, len(y)):
                transitions = set((y[t-1][depth], y[t][depth]) for depth in range(len(y[t])))
                if len(transitions) == 1 and (q_out, q_out) in transitions:
                    # If no other transitions are present, note (O -> O) once.
                    self.A[q_out, q_out] += 1
                else:
                    # Otherwise, note each non-(O -> O) transition, even if it occurs multiple times.
                    for depth, q in enumerate(y[t]):
                        q_prev = y[t-1][depth]
                        if (q_prev, q) == (q_out, q_out):
                            continue
                        self.A[q_prev, q] += 1
                states = set(y[t])
                if len(states) == 1 and q_out in states:
                    # If no other states are present, note `O` once.
                    self.B[q_out, x[t]] += 1
                else:
                    # Otherwise, note each non-`O` state, even if it appears multiple times.
                    for q in y[t]:
                        if q == q_out:
                            continue
                        self.B[q, x[t]] += 1

    def predict(self, sentence_tokens: List[List[str]]) -> List[List[List[str]]]:
        _, X = linguistics.get_n_sentence_token_ids(sentence_tokens, self.v)
        Y = list()

        Q = litbank.ENTITY_LABELS

        for s, x in enumerate(X):
            viterbi = np.zeros((len(Q), len(x)))
            backpointer = np.zeros((len(Q), len(x)), dtype=np.int32)
            for q in range(len(Q)):
                viterbi[q, 0] = self.p[q] * self.B[q, x[0]]
                backpointer[q, 0] = -1
            for t in range(1, len(x)):
                for q in range(len(Q)):
                    candidates = [viterbi[q_prime, t-1] * self.A[q_prime, q] * self.B[q, x[t]]
                                  for q_prime in range(len(Q))]
                    q_max, p_max = max(list(enumerate(candidates)), key=operator.itemgetter(1))
                    viterbi[q, t] = p_max
                    backpointer[q, t] = q_max
            q_max, p_max = max(list(enumerate(viterbi[:, -1])), key=operator.itemgetter(1))
            path = [q_max]
            while len(path) < len(x):
                path.insert(0, backpointer[path[0], len(x) - 1 - len(path)])
            y = [[litbank.ENTITY_LABELS[q]] for q in path]
            Y.append(y)

        return Y

    def save_model(self, dir_):
        pass

    def load_model(self, dir_):
        pass
