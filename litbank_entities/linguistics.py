from typing import List, Optional

import spacy
from spacy.tokens import Doc

from litbank_entities import stats_util

ID_PADDING = 0
ID_OOV = 1

_EXCLUDE_TOKENS_DEFAULT = {'_'}
_CHAR_TO_NORMAL_DEFAULT = {digit: '#' for digit in '0123456789'}
_CHAR_TO_NORMAL_DEFAULT.update({
    '—': '-',  # em-dash
    '“': '"',  # starting and ending quotation marks
    '”': '"',
    '‘': '\'',
    '’': '\'',
    'è': 'e',  # Anglicè (text 49, sentence 11); scène (99, 14)
    'æ': 'ae',  # `æsthetic` (59, 20)
    'ê': 'e',  # fête (61, 54)
    'ö': 'o',  # Ventvögel (74, 32)
    'é': 'e',  # entré (93, 18)
    'ë': 'e',  # aëronaut (95, 10)
})

_FREQ_MIN_DEFAULT = 2

_CHAR_TO_SHAPE = dict()
_CHAR_TO_SHAPE.update({c: 'x' for c in 'abcdefghijklmnopqrstuvwxyz'})
_CHAR_TO_SHAPE.update({c: 'X' for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
_CHAR_TO_SHAPE.update({c: '#' for c in '0123456789'})


def _identity(x):
    return x


def iterate_tokens(sentence_tokens, key=_identity):
    for tokens in sentence_tokens:
        for token in tokens:
            yield key(token)


def get_vocabulary_counts(sentence_tokens: List[List[str]]):
    return stats_util.get_item_counts(iterate_tokens(sentence_tokens))


def iterate_characters(sentence_tokens, key=_identity):
    for tokens in sentence_tokens:
        for token in tokens:
            for c in token:
                yield key(c)


def get_character_counts(sentence_tokens: List[List[str]]):
    return stats_util.get_item_counts(iterate_characters(sentence_tokens))


def process(
        sentence_tokens: List[List[str]],
        sentence_labels: List[List[List[str]]],
        exclude_tokens: Optional[set] = None,
        char_to_normal: Optional[dict] = None,
):
    """
    Remove formatting tokens (underscores) and normalize special characters (diacritics).

    :param sentence_tokens: Tokens per sentence.
    :param sentence_labels: Labels per sentence.
    :param exclude_tokens: Set of tokens to discard. Default is underscore (denotes italics).
    :param char_to_normal: Dict from single characters to normalized strings.
    :return:
    """
    if exclude_tokens is None:
        exclude_tokens = _EXCLUDE_TOKENS_DEFAULT
    if char_to_normal is None:
        char_to_normal = _CHAR_TO_NORMAL_DEFAULT

    # Pre-process; remove excluded tokens (underscores) and normalize characters.
    sentence_tokens_, sentence_labels_ = list(), list()
    for i, tokens in enumerate(sentence_tokens):
        tokens_, labels_ = list(), list()
        for j, token in enumerate(tokens):
            if token in exclude_tokens:
                continue
            tokens_.append(_pre_process_token(token, char_to_normal))
            labels_.append(sentence_labels[i][j])
        sentence_tokens_.append(tokens_)
        sentence_labels_.append(labels_)

    return sentence_tokens_, sentence_labels_


def get_n_sentence_token_ids(sentence_tokens, v, freq_min=_FREQ_MIN_DEFAULT):
    token_to_id = {token: i + 2 for i, (token, count) in enumerate(v) if count >= freq_min}
    return len(token_to_id) + 2, [[token_to_id[token] if token in token_to_id.keys() else ID_OOV
                                   for token in tokens]
                                  for tokens in sentence_tokens]


def _pre_process_token(token: str, char_to_normal: dict):
    token = ''.join(char_to_normal[c] if c in char_to_normal.keys() else c for c in token)
    return token


def get_shape(s):
    return ''.join(_CHAR_TO_SHAPE[c] if c in _CHAR_TO_SHAPE.keys() else c for c in s)


def get_short(s):
    if len(s) == 0:
        return s
    s_ = s[0]
    for c in s[1:]:
        if c != s_[-1]:
            s_ += c
    return s_


def get_nlp():
    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger', 'parser', 'attribute_ruler'])
    nlp.tokenizer = _WhitespaceTokenizer(nlp.vocab)
    return nlp


class _WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        return Doc(self.vocab, words=text.split(' '))


def get_sentence_pos(nlp, sentence_tokens):
    sentences = [' '.join(tokens) for tokens in sentence_tokens]
    return [[(token.pos, token.pos_) for token in doc]
            for doc in nlp.pipe(sentences)]


def get_pos_counts(nlp, sentence_tokens):
    sentence_pos = get_sentence_pos(nlp, sentence_tokens)
    return stats_util.get_item_counts(
        iterate_tokens(sentence_pos),
        key=_identity,
        reverse=False
    )
