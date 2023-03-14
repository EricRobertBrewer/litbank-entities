import os
import pickle
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from litbank_entities import linguistics, litbank, stats_util
from litbank_entities.model import recognizer

CHAR_ID_OOB = 2


class CRFEntityRecognizer(recognizer.EntityRecognizer):

    def __init__(
            self,
            categories,
            nlp,
            window_radius=2,
            char_freq_min=2,
            short_freq_min=3,
            pos_freq_min=3,
            prefix_len=5,
            suffix_len=5,
            batch_size=32,
            epochs=16,
    ):
        super().__init__(categories)
        self.nlp = nlp
        self.window_radius = window_radius
        self.char_freq_min = char_freq_min
        self.short_freq_min = short_freq_min
        self.pos_freq_min = pos_freq_min
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.batch_size = batch_size
        self.epochs = epochs

        self.seq_len = None  # Length of sequences.
        # self.v = None  # Vocabulary and frequencies.
        # self.n = None  # Size of vocabulary, |V|. Includes placeholders for OOV and padding.
        self.c_to_id = None  # Dict of characters to ID.
        self.k_char = None  # Number of distinct values for a single character.
        self.short_to_id = None  #
        self.k_shorts = None
        self.pos_to_id = None
        self.k_pos = None

        self.models = None

    def train(self, sentence_tokens: List[List[str]], sentence_labels: List[List[List[str]]]):
        # sentence_len_max = max(map(len, sentence_tokens))
        # self.seq_len = int(sentence_len_max + 0.05 * sentence_len_max)
        self.seq_len = 166  # Meta-value derived from analysis after splitting large sentences on semi-colon.

        # Words.
        # self.v = linguistics.get_vocabulary_counts(sentence_tokens)
        # self.n, sentence_token_ids = linguistics.get_n_sentence_token_ids(sentence_tokens, self.v)

        # Characters (prefixes and suffixes).
        char_counts = linguistics.get_character_counts(sentence_tokens)
        self.c_to_id = {c: i + 3 for i, (c, count) in enumerate(char_counts) if count >= self.char_freq_min}
        self.k_char = len(self.c_to_id.keys()) + 3  # Supplements: (1) padding, (2) char OOV, (3) index out of bounds.

        # Shape.
        sentence_shapes = [[linguistics.get_shape(token) for token in tokens]
                           for tokens in sentence_tokens]
        sentence_shorts = [[linguistics.get_short(shape) for shape in shapes]
                           for shapes in sentence_shapes]
        short_counts = stats_util.get_item_counts(linguistics.iterate_tokens(sentence_shorts))
        self.short_to_id = {short: i + 2 for i, (short, count) in enumerate(short_counts)
                            if count >= self.short_freq_min}
        self.k_shorts = len(self.short_to_id) + 2

        # Parts of speech.
        pos_counts = linguistics.get_pos_counts(self.nlp, sentence_tokens)
        self.pos_to_id = {pos: i + 2 for i, ((pos, _), count) in enumerate(pos_counts)
                          if count >= self.pos_freq_min}
        self.k_pos = len(self.pos_to_id) + 2

        # Train.
        X = self._get_X(sentence_tokens)

        category_sentence_tags = litbank.get_category_sentence_tags(sentence_labels, self.categories)
        category_sentence_tag_ids = [[[litbank.ENTITY_TAG_TO_ID[tag]
                                       for tag in tags + (['O'] * (self.seq_len - len(tags)))]
                                      for tags in sentence_tags]
                                     for sentence_tags in category_sentence_tags]
        category_Y = [np.array(sentence_tag_ids) for sentence_tag_ids in category_sentence_tag_ids]

        self.models = [self._create_model(X.shape[1:]) for _ in range(len(category_Y))]
        for i_category, (model, Y) in enumerate(zip(self.models, category_Y)):
            print('Fitting category `{}`.'.format(self.categories[i_category]))
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_crf_loss',
                                                          min_delta=2**-6,
                                                          patience=2)]
            model.fit(X, Y, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks, validation_split=1 / 9)

    def predict(self, sentence_tokens: List[List[str]]) -> List[List[List[str]]]:
        X = self._get_X(sentence_tokens)
        category_Y_pred = [model.predict(X, batch_size=self.batch_size) for model in self.models]
        sentence_labels = [[['-'.join([litbank.ENTITY_TAGS[category_Y_pred[k][i][j]], self.categories[k]])
                             if category_Y_pred[k][i][j] != litbank.ENTITY_TAG_TO_ID['O'] else 'O'
                             for k in range(len(category_Y_pred))]
                            for j in range(len(sentence_tokens[i]))]
                           for i in range(len(sentence_tokens))]
        return sentence_labels

    def save_model(self, dir_):
        os.makedirs(dir_, exist_ok=True)
        members = (self.seq_len, self.c_to_id, self.k_char, self.short_to_id, self.k_shorts, self.pos_to_id, self.k_pos)
        with open(os.path.join(dir_, '_members.pickle'), 'wb') as fd:
            pickle.dump(members, fd)
        for i, category in enumerate(self.categories):
            model_path = os.path.join(dir_, 'model_{}'.format(category))
            self.models[i].save(model_path)

    def load_model(self, dir_):
        with open(os.path.join(dir_, '_members.pickle'), 'rb') as fd:
            members = pickle.load(fd)
        self.seq_len, self.c_to_id, self.k_char, self.short_to_id, self.k_shorts, self.pos_to_id, self.k_pos = members
        self.models = list()
        for i, category in enumerate(self.categories):
            model_path = os.path.join(dir_, 'model_{}'.format(category))
            self.models.append(tf.keras.models.load_model(model_path))

    def _pad_and_one_hot(self, sentence_ids, k):
        # Note: Truncation should not be needed if `seq_len` is set to the specially-derived meta-value, 166.
        sentence_ids = [ids + [linguistics.ID_PADDING for _ in range(max(0, self.seq_len - len(ids)))]
                        for ids in sentence_ids]
        return tf.one_hot(sentence_ids, k, axis=-1)

    def _get_character_ids(self, sentence_tokens, i):
        if i >= 0:
            return [[CHAR_ID_OOB if i >= len(token) else
                     self.c_to_id[token[i]] if token[i] in self.c_to_id.keys() else linguistics.ID_OOV
                     for token in tokens]
                    for tokens in sentence_tokens]

        return [[CHAR_ID_OOB if -i - 1 >= len(token) else
                 self.c_to_id[token[i]] if token[i] in self.c_to_id.keys() else linguistics.ID_OOV
                 for token in tokens]
                for tokens in sentence_tokens]

    def _get_X(self, sentence_tokens):
        # _, sentence_token_ids = linguistics.get_n_sentence_token_ids(sentence_tokens, self.v)
        # f_sentence_token_ids = self._pad_and_one_hot(sentence_token_ids, self.n)

        prefix_suffix_indices = tuple(list(range(self.prefix_len)) + list(range(-1, -self.suffix_len - 1, -1)))
        index_sentence_char_ids = [self._get_character_ids(sentence_tokens, i) for i in prefix_suffix_indices]
        f_sentence_char_ids = [self._pad_and_one_hot(sentence_char_ids, self.k_char)
                               for sentence_char_ids in index_sentence_char_ids]

        sentence_shapes = [[linguistics.get_shape(token) for token in tokens]
                           for tokens in sentence_tokens]
        sentence_shorts = [[linguistics.get_short(shape) for shape in shapes]
                           for shapes in sentence_shapes]
        sentence_short_ids = [[self.short_to_id[short] if short in self.short_to_id.keys() else linguistics.ID_OOV
                               for short in shorts]
                              for shorts in sentence_shorts]
        f_sentence_short_ids = self._pad_and_one_hot(sentence_short_ids, self.k_shorts)

        sentence_pos = linguistics.get_sentence_pos(self.nlp, sentence_tokens)
        sentence_pos_ids = [[self.pos_to_id[pos] if pos in self.pos_to_id.keys() else linguistics.ID_OOV
                             for (pos, _) in _pos]
                            for _pos in sentence_pos]
        f_sentence_pos_ids = self._pad_and_one_hot(sentence_pos_ids, self.k_pos)

        # F_self = [f_sentence_token_ids, *f_sentence_char_ids, f_sentence_short_ids, f_sentence_pos_ids]
        F_self = tf.concat([*f_sentence_char_ids, f_sentence_short_ids, f_sentence_pos_ids], 2)
        n, q, k = F_self.shape
        F_lefts = [tf.concat([tf.zeros((n, i, k)), F_self[:, :-i, :]], 1)
                   for i in range(1, self.window_radius + 1)]
        F_rights = [tf.concat([F_self[:, i:, :], tf.zeros((n, i, k))], 1)
                    for i in range(1, self.window_radius + 1)]
        return tf.concat([F_self, *F_lefts, *F_rights], 2)

    def _create_model(self, input_shape):
        # https://github.com/howl-anderson/addons/blob/add_crf_tutorial/docs/tutorials/layers_crf.ipynb
        inputs = tf.keras.Input(shape=input_shape)  # q, k
        base_model = tf.keras.Model(inputs=inputs, outputs=inputs)
        model = tfa.text.crf_wrapper.CRFModelWrapper(base_model, len(litbank.ENTITY_TAGS))
        model.compile(optimizer=tf.keras.optimizers.Adam(0.02))
        return model
