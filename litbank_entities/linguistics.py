import operator
from collections import defaultdict


def get_vocabulary(sentence_tokens):
    token_to_count = defaultdict(int)
    for tokens in sentence_tokens:
        for token in tokens:
            token_to_count[token] += 1
    return sorted(token_to_count.items(), key=operator.itemgetter(1, 0), reverse=True)


def process(sentence_tokens, v):
    return sentence_tokens
