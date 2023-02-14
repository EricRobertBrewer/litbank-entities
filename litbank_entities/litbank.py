import operator
import os
from typing import List, Tuple

SRC_PATH_DEFAULT = 'litbank'

ENTITY_TAGS = ['B-PER', 'B-FAC', 'B-GPE', 'B-LOC', 'B-VEH', 'B-ORG',
               'I-PER', 'I-FAC', 'I-GPE', 'I-LOC', 'I-VEH', 'I-ORG',
               'O']
ENTITY_TAG_TO_INDEX = {tag: i for i, tag in enumerate(ENTITY_TAGS)}
ENTITY_CATEGORIES = [tag[2:] for tag in ENTITY_TAGS[:6]]
ENTITY_CATEGORY_TO_INDEX = {category: i for i, category in enumerate(ENTITY_CATEGORIES)}
ENTITY_CATEGORY_SET = set(ENTITY_CATEGORIES)

Annotation = Tuple[str, str, int, int, str]
Phrase = Tuple[str, int, int, int]


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


def labels_to_phrases(sentence_labels: List[List[List[str]]]) \
        -> List[List[Phrase]]:
    """
    Transform labels (as found in the tsv file) into phrases.

    :param sentence_labels: "Flattened" sentence labels.

    :return: Tuples as `(category, t_start, t_end, nest_depth)` per sentence.
    `t_start` is inclusive and `t_end` is exclusive.
    """
    sentence_phrases = list()
    for labels in sentence_labels:
        phrases = list()
        for depth in range(len(labels[0])):
            category, t_start = None, None
            for t, label in enumerate(labels):
                nest_label = label[depth]
                if nest_label == 'O' or nest_label.startswith('B-'):
                    if category is not None:
                        phrases.append((category, t_start, t, depth))
                        category, t_start = None, None
                    if nest_label.startswith('B-'):
                        category, t_start = nest_label[2:], t
                elif nest_label.startswith('I-'):
                    assert nest_label[2:] == category
                else:
                    raise ValueError('Unexpected label: {}'.format(nest_label))
            if category is not None:
                phrases.append((category, t_start, len(labels), depth))
        phrases.sort(key=operator.itemgetter(1, 2))
        sentence_phrases.append(phrases)
    return sentence_phrases


def _get_lines(path):
    with open(path, 'r') as fd:
        return [line.strip('\n') for line in fd.readlines()]


def _sort_key_file_gutenburg_id(name):
    return int(name[:name.index('_')])
