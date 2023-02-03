import os

import numpy as np


SRC_PATH_DEFAULT = 'litbank'

ENTITY_CATEGORIES = ['PER', 'FAC', 'GPE', 'LOC', 'VEH', 'ORG']
ENTITY_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(ENTITY_CATEGORIES)}
ENTITY_CATEGORY_SET = set(ENTITY_CATEGORIES)


def _sort_key_file_gutenburg_id(name):
    return int(name[:name.index('_')])


def get_ann_paths(src_path=SRC_PATH_DEFAULT):
    entities_brat_path = os.path.join(src_path, 'entities', 'brat')
    brat_files = sorted(os.listdir(entities_brat_path), key=_sort_key_file_gutenburg_id)
    return [os.path.join(entities_brat_path, file) for file in brat_files if file.endswith('.ann')]


def get_txt_paths(src_path=SRC_PATH_DEFAULT):
    entities_brat_path = os.path.join(src_path, 'entities', 'brat')
    brat_files = sorted(os.listdir(entities_brat_path), key=_sort_key_file_gutenburg_id)
    return [os.path.join(entities_brat_path, file) for file in brat_files if file.endswith('.txt')]


def get_tsv_paths(src_path=SRC_PATH_DEFAULT):
    entities_tsv_path = os.path.join(src_path, 'entities', 'tsv')
    tsv_files = sorted(os.listdir(entities_tsv_path), key=_sort_key_file_gutenburg_id)
    return [os.path.join(entities_tsv_path, file) for file in tsv_files]


def get_text_annotations(src_path=SRC_PATH_DEFAULT):
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


def get_texts(src_path=SRC_PATH_DEFAULT):
    texts = list()
    txt_paths = get_txt_paths(src_path=src_path)
    for txt_path in txt_paths:
        txt_lines = _get_lines(txt_path)
        text = ' '.join(txt_lines)
        texts.append(text)
    return texts


def get_text_sentence_tokens_labels(src_path=SRC_PATH_DEFAULT):
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
            sentence_tokens.append(np.array(tokens))
            sentence_labels.append(np.array(labels))
        text_sentence_tokens.append(sentence_tokens)
        text_sentence_labels.append(sentence_labels)
    return text_sentence_tokens, text_sentence_labels


def _get_lines(path):
    with open(path, 'r') as fd:
        return [line.strip('\n') for line in fd.readlines()]
