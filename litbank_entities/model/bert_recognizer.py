import os
from typing import List

import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification

from litbank_entities import litbank
from litbank_entities.model import recognizer

PRETRAINED_NAME = 'distilbert-base-cased'

PIECE_CLS = '[CLS]'
PIECE_SEP = '[SEP]'
PIECE_PAD = '[PAD]'


def create_model_resources():
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_NAME)
    return tokenizer,


class BertEntityRecognizer(recognizer.EntityRecognizer):

    def __init__(self, categories, *resources, lr=2e-5, epochs=2, batch_size=8):
        if len(categories) != 1:
            raise ValueError('Categories has to be 1 right now!')
        self.category = categories[0]
        super().__init__(categories)

        tokenizer, = resources
        self.tokenizer = tokenizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.seq_len = 166 + 2  # Based on analysis of entire data set.

        self.model = None

    def train(self, sentence_tokens: List[List[str]], sentence_labels: List[List[List[str]]]):
        # Encode tokens in word pieces.
        encodings, offset_mapping = self.tokenize(sentence_tokens)

        # Encode and expand tags to align with word pieces.
        category_sentence_tags = litbank.get_category_sentence_tags(sentence_labels, self.categories)
        sentence_tags = category_sentence_tags[0]  # TODO: Remove to enable multiple categories.
        sentence_tag_ids = [[litbank.ENTITY_TAG_TO_ID[tag] for tag in tags]
                            for tags in sentence_tags]
        sentence_tag_pieces = expand_tag_ids(sentence_tag_ids, offset_mapping)

        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            sentence_tag_pieces
        ))

        self.model = TFDistilBertForTokenClassification.from_pretrained(
            PRETRAINED_NAME,
            num_labels=len(litbank.ENTITY_TAGS)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=optimizer)  # , loss=self.model.compute_loss)
        history = self.model.fit(dataset.shuffle(1024), epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, sentence_tokens: List[List[str]]) -> List[List[List[str]]]:
        encodings, offset_mapping = self.tokenize(sentence_tokens, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings),))
        sentence_tag_piece_preds = self.model.predict(dataset, batch_size=self.batch_size)

        # Contract tag IDs (that are expanded by the model).
        sentence_labels = list()
        sentence_tag_pieces = tf.math.argmax(sentence_tag_piece_preds.logits, axis=-1)
        for tag_pieces, offsets in zip(sentence_tag_pieces, offset_mapping):
            labels = list()
            for tag_piece, offset in zip(tag_pieces, offsets):
                if offset[1] == 0 or offset[0] != 0:
                    continue  # TODO: Verify that remaining tag pieces agree with first (or vote).
                if tag_piece != litbank.ENTITY_TAG_TO_ID['O']:
                    nest_labels = ['-'.join((litbank.ENTITY_TAGS[tag_piece], self.category))]
                else:
                    nest_labels = ['O']
                labels.append(nest_labels)
            sentence_labels.append(labels)
        return sentence_labels

    def tokenize(self, sentence_tokens, return_tensors=None):
        # https://huggingface.co/transformers/v3.5.1/custom_datasets.html#token-classification-with-w-nut-emerging-entities
        encodings = self.tokenizer(
            sentence_tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors=return_tensors
        )
        offset_mapping = encodings.offset_mapping
        encodings.pop('offset_mapping')
        return encodings, offset_mapping

    def save_model(self, dir_):
        model_path = os.path.join(dir_, 'model_{}'.format(self.category))
        self.model.save(model_path)

    def load_model(self, dir_):
        model_path = os.path.join(dir_, 'model_{}'.format(self.category))
        self.model = tf.keras.models.load_model(model_path)
        

def expand_tag_ids(sentence_tag_ids, offset_mapping, fill_id=litbank.ENTITY_TAG_TO_ID['O']):
    sentence_tag_pieces = list()
    for tag_ids, offsets in zip(sentence_tag_ids, offset_mapping):
        tag_pieces = list()
        tag_i = -1
        for offset in offsets:
            if offset[1] == 0:
                # Special token.
                tag_pieces.append(fill_id)
            elif offset[0] != 0:
                # Continuation of a word. Also continue the last-used tag.
                tag_pieces.append(tag_pieces[-1])
            else:
                # The start of a new word - use a new tag.
                tag_i += 1
                tag_pieces.append(tag_ids[tag_i])
        sentence_tag_pieces.append(tag_pieces)
    return sentence_tag_pieces
