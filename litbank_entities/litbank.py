import operator
import os
from typing import List, Tuple

SRC_PATH_DEFAULT = 'litbank'

ENTITY_TAGS = ['B', 'I', 'O']
ENTITY_CATEGORIES = ['PER', 'FAC', 'GPE', 'LOC', 'VEH', 'ORG']
ENTITY_CATEGORY_SET = set(ENTITY_CATEGORIES)
ENTITY_LABELS = ['{}-{}'.format(tag, category) for tag in ENTITY_TAGS[:2] for category in ENTITY_CATEGORIES] + ['O']
ENTITY_TAG_TO_ID = {tag: i for i, tag in enumerate(ENTITY_TAGS)}
ENTITY_CATEGORY_TO_ID = {category: i for i, category in enumerate(ENTITY_CATEGORIES)}
ENTITY_LABEL_TO_ID = {label: i for i, label in enumerate(ENTITY_LABELS)}

Annotation = Tuple[str, str, int, int, str]
Phrase = Tuple[int, int, int]


def get_ann_paths(src_path=SRC_PATH_DEFAULT) -> List[str]:
    entities_brat_path = os.path.join(src_path, 'entities', 'brat')
    brat_files = sorted(os.listdir(entities_brat_path), key=_sort_key_file_gutenburg_id)
    return [os.path.join(entities_brat_path, file) for file in brat_files if file.endswith('.ann')]


def get_txt_paths(src_path=SRC_PATH_DEFAULT) -> List[str]:
    entities_brat_path = os.path.join(src_path, 'entities', 'brat')
    brat_files = sorted(os.listdir(entities_brat_path), key=_sort_key_file_gutenburg_id)
    return [os.path.join(entities_brat_path, file) for file in brat_files if file.endswith('.txt')]


def get_tsv_paths(src_path=SRC_PATH_DEFAULT) -> List[str]:
    entities_tsv_path = os.path.join(src_path, 'entities', 'tsv')
    tsv_files = sorted(os.listdir(entities_tsv_path), key=_sort_key_file_gutenburg_id)
    return [os.path.join(entities_tsv_path, file) for file in tsv_files]


def get_text_annotations(src_path=SRC_PATH_DEFAULT) -> List[List[Annotation]]:
    text_annotations = list()
    ann_paths = get_ann_paths(src_path=src_path)
    for ann_path in ann_paths:
        annotations = list()
        ann_lines = _get_lines(ann_path)
        for line in ann_lines:
            id_, ann, phrase = line.split('\t')
            category, a, b = ann.split(' ')
            annotations.append((id_, category, int(a), int(b), phrase))
        text_annotations.append(annotations)
    return text_annotations


def get_text_lines(src_path=SRC_PATH_DEFAULT) -> List[List[str]]:
    text_lines = list()
    txt_paths = get_txt_paths(src_path=src_path)
    for txt_path in txt_paths:
        lines = _get_lines(txt_path)
        text_lines.append(lines)
    return text_lines


def get_text_sentence_tokens_labels(src_path=SRC_PATH_DEFAULT) \
        -> Tuple[List[List[List[str]]], List[List[List[List[str]]]]]:
    text_sentence_tokens = list()  # (100, |S|, |T|)
    text_sentence_labels = list()  # (100, |S|, |T|, N)
    tsv_paths = get_tsv_paths(src_path=src_path)
    for tsv_path in tsv_paths:
        sentence_tokens = list()
        tokens = list()
        sentence_labels = list()
        labels = list()
        tsv_lines = _get_lines(tsv_path)
        for tsv_line in tsv_lines:
            if tsv_line == '':
                sentence_tokens.append(tokens)
                tokens = list()
                sentence_labels.append(labels)
                labels = list()
            else:
                token, *label = tsv_line.strip('\t').split('\t')
                tokens.append(token)
                labels.append(label)
        if len(tokens) > 0:
            sentence_tokens.append(tokens)
            sentence_labels.append(labels)
        text_sentence_tokens.append(sentence_tokens)
        text_sentence_labels.append(sentence_labels)
    return text_sentence_tokens, text_sentence_labels


def flatten_texts(text_sentence_tokens: List[List[List[str]]],
                  text_sentence_labels: List[List[List[List[str]]]]) \
        -> Tuple[List[List[str]], List[List[List[str]]]]:
    sentence_tokens, sentence_labels = list(), list()
    for i in range(len(text_sentence_tokens)):
        for tokens in text_sentence_tokens[i]:
            sentence_tokens.append(tokens)
        for labels in text_sentence_labels[i]:
            sentence_labels.append(labels)
    return sentence_tokens, sentence_labels


def get_category_sentence_phrases(sentence_labels: List[List[List[str]]], categories=None) \
        -> List[List[List[Phrase]]]:
    """
    Transform labels (as found in the tsv file) into phrases.

    :param sentence_labels: "Flattened" sentence labels.
    :param categories: Iterable of category names.

    :return: Tuples as `(t_start, t_end, nest_depth)` per sentence per category as
    ordered in `ENTITY_CATEGORIES`. `t_start` is inclusive and `t_end` is exclusive.
    """
    if categories is None:
        categories = ENTITY_CATEGORIES
    category_to_id = {category: i for i, category in enumerate(categories)}

    category_sentence_phrases = [list() for _ in categories]
    for labels in sentence_labels:
        category_phrases = [list() for _ in categories]
        for depth in range(len(labels[0])):
            category_id, t_start = None, None
            for t, label in enumerate(labels):
                nest_label = label[depth]
                if nest_label == 'O' or nest_label.startswith('B-'):
                    if category_id is not None:
                        # noinspection PyTypeChecker
                        category_phrases[category_id].append((t_start, t, depth))
                        category_id, t_start = None, None
                    if nest_label.startswith('B-') and nest_label[2:] in category_to_id.keys():
                        category_id, t_start = category_to_id[nest_label[2:]], t
                elif nest_label.startswith('I-'):
                    # assert category_to_id[nest_label[2:]] == category_id
                    pass  # If above assertion fails, the recognizer emitted an extraneous `I`.
                else:
                    raise ValueError('Unexpected label: {}'.format(nest_label))
            if category_id is not None:
                # noinspection PyTypeChecker
                category_phrases[category_id].append((t_start, len(labels), depth))
        for i, phrases in enumerate(category_phrases):
            phrases.sort(key=operator.itemgetter(0, 1))
            category_sentence_phrases[i].append(phrases)
    return category_sentence_phrases


def get_category_sentence_tags(sentence_labels: List[List[List[str]]], categories=None):
    if categories is None:
        categories = ENTITY_CATEGORIES

    category_sentence_phrases = get_category_sentence_phrases(sentence_labels, categories=categories)
    category_sentence_tags = [[['O' for _ in range(len(labels))]
                               for labels in sentence_labels]
                              for _ in range(len(categories))]
    for category_id, sentence_phrases in enumerate(category_sentence_phrases):
        for i_sentence, phrases in enumerate(sentence_phrases):
            j = 0
            while j < len(phrases):
                # Collect overlapping phrases.
                candidates = [phrases[j]]
                while j + 1 < len(phrases) and phrases[j + 1][0] < phrases[j][1]:
                    candidates.append(phrases[j + 1])
                    j += 1
                # Pick the shortest when two or more phrases overlap.
                winner = min(candidates, key=lambda phrase: phrase[1] - phrase[0])
                # Edit the tags according to this phrase.
                t_start, t_end, _ = winner
                category_sentence_tags[category_id][i_sentence][t_start] = 'B'
                for t in range(t_start + 1, t_end):
                    category_sentence_tags[category_id][i_sentence][t] = 'I'
                j += 1
    return category_sentence_tags


def split_large_sentences(sentence_tokens, sentence_labels, separators=(';',), max_len=161):
    sentence_tokens_, sentence_labels_ = list(), list()
    for i_sentence, tokens in enumerate(sentence_tokens):
        labels = sentence_labels[i_sentence]
        if len(tokens) <= max_len:
            sentence_tokens_.append(tokens)
            sentence_labels_.append(labels)
            continue
        for separator in separators:
            index = _get_index_near_middle(tokens, labels, separator)
            if index != -1:
                sentence_tokens_.append(tokens[:index + 1])
                sentence_labels_.append(labels[:index + 1])
                sentence_tokens_.append(tokens[index + 1:])
                sentence_labels_.append(labels[index + 1:])
                break
    return sentence_tokens_, sentence_labels_


def _get_lines(path):
    with open(path, 'r') as fd:
        return [line.strip('\n') for line in fd.readlines()]


def _sort_key_file_gutenburg_id(name):
    return int(name[:name.index('_')])


def _get_index_near_middle(tokens, labels, separator):
    mid = len(tokens) // 2
    indices = [mid]
    for d in range(1, mid):
        indices.append(mid - d)
        indices.append(mid + d)
    indices.append(0)
    if len(tokens) % 2 == 0:
        indices.append(len(tokens) - 1)
    for index in indices:
        if tokens[index] == separator and all(nest_label == 'O' for nest_label in labels[index]):
            return index
    return -1
