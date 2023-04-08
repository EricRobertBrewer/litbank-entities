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
    parser.add_argument('--test_ratio',
                        type=float,
                        required=False,
                        help='Ratio of test data for a single evaluation.')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='Seed for random number generator.')
    parser.add_argument('--categories',
                        default='PER,FAC,GPE,LOC,VEH,ORG',
                        type=str,
                        help='Comma-separated category names, e.g., `PER,LOC`.')
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
        test_ratio=args.test_ratio,
        seed=args.seed,
        category_set=set(args.categories.split(',')),
        keep_ratio=args.keep_ratio)


def run(classname, folds=10, test_ratio=None, seed=1, category_set=None, keep_ratio=1.0):
    if test_ratio is None:
        if folds < 2:
            raise ValueError('Unexpected number of folds: {:d}'.format(folds))
    else:
        if test_ratio <= 0.0 or test_ratio >= 1.0:
            raise ValueError('Unexpected test ratio: {:f}'.format(test_ratio))
    rng = np.random.default_rng(seed=seed)
    if category_set is None:
        category_set = litbank.ENTITY_CATEGORY_SET

    timestamp = int(time.time())
    class_dir = os.path.join(OUTPUT_DIR, classname)
    os.makedirs(class_dir, exist_ok=True)
    params_path = os.path.join(class_dir, '{:d}-params.txt'.format(timestamp))
    with open(params_path, 'w') as fd:
        fd.write('classname: {}\n'.format(classname))
        if test_ratio is None:
            fd.write('folds: {:d}\n'.format(folds))
        else:
            fd.write('test_ratio: {:.2f}\n'.format(test_ratio))
        fd.write('seed: {:d}\n'.format(seed))
        fd.write('category_set: {}\n'.format(str(category_set)))
        fd.write('keep_ratio: {:.4f}\n'.format(keep_ratio))

    text_sentence_tokens, text_sentence_labels = litbank.get_text_sentence_tokens_labels()

    if DEBUG:
        # Emulate each sentence from the first text as belonging to separate texts.
        text_sentence_tokens = [[tokens] for tokens in text_sentence_tokens[0]]
        text_sentence_labels = [[labels] for labels in text_sentence_labels[0]]
        test_ratio = 1 / 5

    fold_datasets = get_fold_datasets(text_sentence_tokens, text_sentence_labels, folds, test_ratio, rng)

    # Evaluate for each fold.
    fold_category_counts, fold_category_metrics = list(), list()
    categories = [category for category in litbank.ENTITY_CATEGORIES if category in category_set]
    resources = create_model_resources(classname)
    for fold, (train_instances, test_instances) in enumerate(fold_datasets):
        print('Starting fold {:d} / {:d}.'.format(fold + 1, len(fold_datasets)))
        train_pairs = list()
        for pair in zip(*train_instances):
            if keep_ratio > rng.uniform():
                train_pairs.append(pair)
        train_instances = zip(*train_pairs)
        model = create_model(classname, categories, resources)
        category_counts, category_metrics = evaluate(model, train_instances, test_instances, categories)
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


def get_fold_datasets(text_sentence_tokens, text_sentence_labels, folds, test_ratio, rng):
    fold_datasets = list()
    text_sentence_pairs = list(zip(text_sentence_tokens, text_sentence_labels))
    rng.shuffle(text_sentence_pairs)
    n = len(text_sentence_pairs)
    if test_ratio is None:
        # Cross-validation.
        for i in range(folds):
            train_text_instances, test_text_instances = \
                get_split_instances(text_sentence_pairs, int(n / folds * i), int(n / folds * (i + 1)))
            train_instances = litbank.flatten_texts(*train_text_instances)
            test_instances = litbank.flatten_texts(*test_text_instances)
            fold_datasets.append((train_instances, test_instances))
    else:
        # Single evaluation.
        train_text_instances, test_text_instances = \
            get_split_instances(text_sentence_pairs, 0, int(n * test_ratio))
        train_instances = litbank.flatten_texts(*train_text_instances)
        test_instances = litbank.flatten_texts(*test_text_instances)
        fold_datasets.append((train_instances, test_instances))

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
    train_instances = linguistics.process(*train_instances)
    train_instances = litbank.split_large_sentences(*train_instances)
    test_instances = linguistics.process(*test_instances)
    test_instances = litbank.split_large_sentences(*test_instances)
    test_sentence_tokens, test_sentence_labels = test_instances
    model.train(*train_instances)
    test_sentence_preds = model.predict(test_sentence_tokens)
    return pm.get_phrase_counts_and_metrics(test_sentence_labels, test_sentence_preds, categories)


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
