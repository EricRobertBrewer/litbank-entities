import operator
from collections import defaultdict

TOKEN_OOV = '<OOV>'


def get_vocabulary(sentence_tokens):
    token_to_count = defaultdict(int)
    token_to_count[TOKEN_OOV] = 0
    for tokens in sentence_tokens:
        for token in tokens:
            token_to_count[token] += 1
    return sorted(token_to_count.items(), key=operator.itemgetter(1, 0), reverse=True)


def process(sentence_tokens, freq_min=1):
    v = get_vocabulary(sentence_tokens)
    token_to_count = dict(v)
    sentence_tokens_ = list()
    for tokens in sentence_tokens:
        tokens_ = list()
        for token in tokens:
            if token_to_count[token] <= freq_min:
                token_ = TOKEN_OOV
            else:
                token_ = token
            tokens_.append(token_)
        sentence_tokens_.append(tokens_)
    return sentence_tokens_
