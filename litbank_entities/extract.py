import argparse
import os
import time

import numpy as np

from litbank_entities import linguistics, litbank, phrase_metrics as pm
from litbank_entities.model import crf_recognizer, hmm_recognizer, bert_recognizer, zero_r_recognizer

DEBUG = False

OUTPUT_DIR = 'output'


def main():
    global DEBUG
    parser = argparse.ArgumentParser(
        description='Perform named-entity recognition over `litbank` text samples.'
    )
    parser.add_argument('classname',
                        choices=['bert', 'crf', 'hmm', 'zero'],
                        help='Name of the model.')
    parser.add_argument('--folds',
                        default=10,
                        type=int,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--fold',
                        type=int,
                        required=False,
                        help='Fold number for cross-evaluation when folds have to be done in separate processes.')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='Seed for random number generator.')
    parser.add_argument('--categories',
                        default='PER,FAC,GPE,LOC,VEH,ORG',
                        type=str,
                        help='Comma-separated category names, e.g., `PER,LOC`.')
    parser.add_argument('--skip_processing',
                        action='store_true',
                        help='Flag to leave text un-processed (such as digits -> `#`).')
    parser.add_argument('--keep_ratio',
                        default=1.0,
                        type=float,
                        help='Portion of training data to use.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Flag to indicate debug mode.')

    args = parser.parse_args()
    DEBUG = args.debug
    run(
        args.classname,
        folds=args.folds,
        fold=args.fold,
        seed=args.seed,
        category_set=set(args.categories.split(',')),
        skip_processing=args.skip_processing,
        keep_ratio=args.keep_ratio)


def run(classname, folds=10, fold=None, seed=1, category_set=None, skip_processing=False, keep_ratio=1.0):
    if folds < 2:
        raise ValueError('Unexpected number of folds: {:d}'.format(folds))
    if fold is not None:
        if fold < 1 or fold > folds:
            raise ValueError('Unexpected fold, given {:d} folds: {:d}'.format(folds, fold))

    rng = np.random.default_rng(seed=seed)

    if category_set is None:
        category_set = litbank.ENTITY_CATEGORY_SET
    categories = [category for category in litbank.ENTITY_CATEGORIES if category in category_set]

    timestamp = int(time.time())
    class_dir = os.path.join(OUTPUT_DIR, classname)
    os.makedirs(class_dir, exist_ok=True)
    params_path = os.path.join(class_dir, '{:d}-params.txt'.format(timestamp))
    with open(params_path, 'w') as fd:
        fd.write('classname: {}\n'.format(classname))
        fd.write('folds: {:d}\n'.format(folds))
        if fold is not None:
            fd.write('fold: {:d}\n'.format(fold))
        fd.write('seed: {:d}\n'.format(seed))
        fd.write('category_set: {}\n'.format(str(category_set)))
        fd.write('skip_processing: {}\n'.format(str(skip_processing)))
        fd.write('keep_ratio: {:.4f}\n'.format(keep_ratio))

    text_sentence_tokens, text_sentence_labels = litbank.get_text_sentence_tokens_labels()

    if DEBUG:
        # Emulate each sentence from the first text as belonging to separate texts.
        text_sentence_tokens = [[tokens] for tokens in text_sentence_tokens[0]]
        text_sentence_labels = [[labels] for labels in text_sentence_labels[0]]

    fold_datasets = get_fold_datasets(text_sentence_tokens, text_sentence_labels, folds, fold, rng, skip_processing)

    # Evaluate for each fold.
    fold_category_counts, fold_category_metrics = list(), list()
    resources = create_model_resources(classname)
    for i, (train_instances, test_instances) in enumerate(fold_datasets):
        fold_ = i + 1 if fold is None else fold
        print('Starting fold {:d} / {:d}.'.format(fold_, folds))
        train_pairs = list()
        for pair in zip(*train_instances):
            if keep_ratio > rng.uniform():
                train_pairs.append(pair)
        train_instances = zip(*train_pairs)
        model = create_model(classname, categories, resources)
        category_counts, category_metrics, test_sentence_preds = \
            evaluate(model, train_instances, test_instances, categories)
        write_instances(test_instances, test_sentence_preds, categories, timestamp, fold_, class_dir)
        fold_category_counts.append(category_counts)
        fold_category_metrics.append(category_metrics)

    # Calculate macro- and micro- averages.
    results_path = os.path.join(class_dir, '{:d}-results.txt'.format(timestamp))
    with open(results_path, 'w') as fd:
        category_metrics_macro, category_metrics_micro = \
            pm.calculate_fold_averages(fold_category_counts, fold_category_metrics)
        for level, category_metrics in zip(('Macro', 'Micro'), (category_metrics_macro, category_metrics_micro)):
            fd.write('{}:\n'.format(level))
            write_results(category_metrics, categories, fd)


def get_fold_datasets(text_sentence_tokens, text_sentence_labels, folds, fold, rng, skip_processing):
    fold_datasets = list()

    def _process_and_append(_train_text_instances, _test_text_instances):
        nonlocal fold_datasets
        _train_instances = litbank.flatten_texts(*_train_text_instances)
        _test_instances = litbank.flatten_texts(*_test_text_instances)
        if not skip_processing:
            _train_instances = linguistics.process(*_train_instances)
            _test_instances = linguistics.process(*_test_instances)
        _train_instances = litbank.split_large_sentences(*_train_instances)
        _test_instances = litbank.split_large_sentences(*_test_instances)
        fold_datasets.append((_train_instances, _test_instances))

    text_sentence_pairs = list(zip(text_sentence_tokens, text_sentence_labels))
    rng.shuffle(text_sentence_pairs)
    n = len(text_sentence_pairs)
    if fold is None:
        # Cross-validation.
        for i in range(folds):
            train_text_instances, test_text_instances = \
                get_split_instances(text_sentence_pairs, int(n / folds * i), int(n / folds * (i + 1)))
            _process_and_append(train_text_instances, test_text_instances)
    else:
        # Single evaluation.
        train_text_instances, test_text_instances = \
            get_split_instances(text_sentence_pairs, int(n / folds * (fold - 1)), int(n / folds * fold))
        _process_and_append(train_text_instances, test_text_instances)

    return fold_datasets


def get_split_instances(pairs, test_start, test_end):
    train_pairs = pairs[:test_start] + pairs[test_end:]
    test_pairs = pairs[test_start:test_end]
    return list(zip(*train_pairs)), list(zip(*test_pairs))


def create_model_resources(classname):
    if classname == 'zero':
        return tuple()
    if classname == 'hmm':
        return tuple()
    if classname == 'crf':
        nlp = linguistics.get_nlp()
        return nlp,
    if classname == 'bert':
        return bert_recognizer.create_model_resources()
    raise ValueError('Unexpected model class name: {}'.format(classname))


def create_model(classname, categories, resources):
    if classname == 'zero':
        return zero_r_recognizer.ZeroREntityRecognizer(categories)
    if classname == 'hmm':
        return hmm_recognizer.HMMEntityRecognizer(categories, ignore_nested=True)
    if classname == 'crf':
        kwargs = dict()
        if DEBUG:
            kwargs['epochs'] = 1
        return crf_recognizer.CRFEntityRecognizer(categories, *resources, **kwargs)
    if classname == 'bert':
        kwargs = dict()
        if DEBUG:
            kwargs['epochs'] = 1
        return bert_recognizer.BertEntityRecognizer(categories, *resources, **kwargs)
    raise ValueError('Unexpected model class name: {}'.format(classname))


def evaluate(model, train_instances, test_instances, categories):
    test_sentence_tokens, test_sentence_labels = test_instances
    model.train(*train_instances)
    test_sentence_preds = model.predict(test_sentence_tokens)
    category_counts, category_metrics = \
        pm.get_phrase_counts_and_metrics(test_sentence_labels, test_sentence_preds, categories)
    return category_counts, category_metrics, test_sentence_preds


def write_instances(instances, sentence_preds, categories, timestamp, fold, class_dir):
    sentence_tokens, sentence_labels = instances
    category_sentence_label_phrases = litbank.get_category_sentence_phrases(sentence_labels, categories)
    category_sentence_pred_phrases = litbank.get_category_sentence_phrases(sentence_preds, categories)
    for k, category in enumerate(categories):
        sentence_label_phrases = category_sentence_label_phrases[k]
        sentence_pred_phrases = category_sentence_pred_phrases[k]
        instances_fname = '{:d}-instances-{}-{:d}.txt'.format(timestamp, category, fold)
        instances_path = os.path.join(class_dir, instances_fname)
        with open(instances_path, 'w') as fd:
            fd.write('{:d}\n'.format(len(sentence_tokens)))
            for i, tokens in enumerate(sentence_tokens):
                label_phrases = sentence_label_phrases[i]
                pred_phrases = sentence_pred_phrases[i]
                fd.write('\n')
                fd.write('{:d}|{}\n'.format(len(tokens), ' '.join(tokens)))
                fd.write('{:d}\n'.format(len(label_phrases)))
                for phrase in label_phrases:
                    start, end = phrase[:2]
                    fd.write('{:d}|{:d}|{}\n'.format(start, end, ' '.join(tokens[start:end])))
                fd.write('{:d}\n'.format(len(pred_phrases)))
                for phrase in pred_phrases:
                    start, end = phrase[:2]
                    fd.write('{:d}|{:d}|{}\n'.format(start, end, ' '.join(tokens[start:end])))


def write_results(category_metrics, categories, fd):
    metric_names = (pm.METRIC_PRECISION, pm.METRIC_RECALL, pm.METRIC_F1)
    coverages = (pm.COVERAGE_EXACT, pm.COVERAGE_PARTIAL)

    for category_id, metrics in enumerate(category_metrics):
        fd.write('{}\n'.format(categories[category_id]))
        fd.write('\t'.join(('', *(metric.capitalize() for metric in metric_names))))
        fd.write('\n')
        for coverage in coverages:
            values = (metrics[metric][coverage] for metric in metric_names)
            fd.write('\t'.join((coverage.capitalize(), *('{:.4f}'.format(value) for value in values))))
            fd.write('\n')
        fd.write('\n')


if __name__ == '__main__':
    main()
