import argparse

import numpy as np

from litbank_entities import litbank, metrics
from litbank_entities.model import hmm_recognizer, zero_r_recognizer


def main():
    parser = argparse.ArgumentParser(
        description='Perform named-entity recognition over `litbank` text samples.'
    )
    parser.add_argument('classname',
                        choices=['hmm', 'zero'],
                        help='Name of the model.')
    parser.add_argument('--folds',
                        '-F',
                        default=10,
                        type=int,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--seed',
                        '-S',
                        default=1,
                        type=int,
                        help='Seed for random number generator.')
    parser.add_argument('--mix_texts',
                        '-M',
                        action='store_true',
                        help='Flag to allow sampling of train and test instances from the same text.')

    args = parser.parse_args()
    classname = args.classname
    seed = args.seed
    folds = args.folds
    mix_texts = args.mix_texts
    cross_validation(classname, folds=folds, seed=seed, mix_texts=mix_texts)


def cross_validation(classname, folds=10, seed=1, mix_texts=False):
    if folds < 2:
        raise ValueError('Unexpected number of folds: {:d}'.format(folds))
    if seed is not None:
        np.random.seed(seed)

    text_sentence_tokens, text_sentence_labels = litbank.get_text_sentence_tokens_labels()
    fold_datasets = list()
    if mix_texts:
        # Allow text mixing; flatten texts first.
        sentence_tokens, sentence_labels = litbank.flatten_texts(text_sentence_tokens, text_sentence_labels)
        n = len(sentence_tokens)
        indices = np.random.permutation(n)
        for i in range(folds):
            test_start, test_end = int(n / folds * i), int(n / folds * (i + 1))
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate((indices[:test_start], indices[test_end:]))
            test_sentence_tokens = [sentence_tokens[index] for index in test_indices]
            test_sentence_labels = [sentence_labels[index] for index in test_indices]
            train_sentence_tokens = [sentence_tokens[index] for index in train_indices]
            train_sentence_labels = [sentence_labels[index] for index in train_indices]
            test_instances = test_sentence_tokens, test_sentence_labels
            train_instances = train_sentence_tokens, train_sentence_labels
            fold_datasets.append((train_instances, test_instances))
    else:
        # Flatten texts after splitting data.
        n = len(text_sentence_tokens)
        indices = np.random.permutation(n)
        for i in range(folds):
            test_start, test_end = int(n / folds * i), int(n / folds * (i + 1))
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate((indices[:test_start], indices[test_end:]))
            test_text_sentence_tokens = [text_sentence_tokens[index] for index in test_indices]
            test_text_sentence_labels = [text_sentence_labels[index] for index in test_indices]
            train_text_sentence_tokens = [text_sentence_tokens[index] for index in train_indices]
            train_text_sentence_labels = [text_sentence_labels[index] for index in train_indices]
            test_instances = litbank.flatten_texts(test_text_sentence_tokens, test_text_sentence_labels)
            train_instances = litbank.flatten_texts(train_text_sentence_tokens, train_text_sentence_labels)
            fold_datasets.append((train_instances, test_instances))

    # Evaluate for each fold.
    fold_category_to_counts, fold_category_to_metrics = list(), list()
    for train_instances, test_instances in fold_datasets:
        model = create_model(classname)
        category_to_counts, category_to_metrics = evaluate(model, train_instances, test_instances)
        fold_category_to_counts.append(category_to_counts)
        fold_category_to_metrics.append(category_to_metrics)

    # Calculate macro- and micro- averages.
    category_to_metrics_macro, category_to_metrics_micro = \
        metrics.calculate_fold_averages(fold_category_to_counts, fold_category_to_metrics)
    for level, category_to_metrics in zip(('Macro', 'Micro'), (category_to_metrics_macro, category_to_metrics_micro)):
        print('{}:'.format(level))
        print_metrics_console(category_to_metrics)


def create_model(classname):
    if classname == 'zero':
        return zero_r_recognizer.ZeroREntityRecognizer()
    elif classname == 'hmm':
        return hmm_recognizer.HMMEntityRecognizer(ignore_nested=True)
    raise ValueError('Unexpected model class name: {}'.format(classname))


def evaluate(model, train_instances, test_instances):
    train_sentence_tokens, train_sentence_labels = train_instances
    test_sentence_tokens, test_sentence_labels = test_instances
    model.train(train_sentence_tokens, train_sentence_labels)
    test_sentence_preds = model.predict(test_sentence_tokens)
    return metrics.get_phrase_counts_and_metrics(test_sentence_labels, test_sentence_preds)


def print_metrics_console(category_to_metrics):
    metrics_ = (metrics.METRIC_PRECISION, metrics.METRIC_RECALL, metrics.METRIC_F1)
    coverages = (metrics.COVERAGE_EXACT, metrics.COVERAGE_PARTIAL)

    for category in litbank.ENTITY_CATEGORIES:
        print('{}'.format(category))
        print('\t'.join(('', *(metric.capitalize() for metric in metrics_))))
        for coverage in coverages:
            values = (category_to_metrics[category][metric][coverage] for metric in metrics_)
            print('\t'.join((coverage.capitalize(), *('{:.4f}'.format(value) for value in values))))
        print()


if __name__ == '__main__':
    main()
