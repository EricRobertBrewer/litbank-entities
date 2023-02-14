from typing import List, Tuple

from litbank_entities import litbank

COUNT_TRUE_POSITIVE = 'tp'  # Numerator.
COUNT_RETRIEVED = 'retrieved'  # Denominator for precision.
COUNT_RELEVANT = 'relevant'  # Denominator for recall.

METRIC_PRECISION = 'precision'
METRIC_RECALL = 'recall'
METRIC_F1 = 'f1'

COVERAGE_EXACT = 'exact'
COVERAGE_PARTIAL = 'partial'


def _create_category_to_counts():
    category_to_counts = dict()
    for category in litbank.ENTITY_CATEGORIES:
        counts = dict()
        counts[COUNT_TRUE_POSITIVE] = {coverage: 0 for coverage in (COVERAGE_EXACT, COVERAGE_PARTIAL)}
        for name in (COUNT_RETRIEVED, COUNT_RELEVANT):
            counts[name] = 0
        category_to_counts[category] = counts
    return category_to_counts


def _create_category_to_metrics():
    return {category: {name: {coverage: 0
                              for coverage in (COVERAGE_EXACT, COVERAGE_PARTIAL)}
                       for name in (METRIC_PRECISION, METRIC_RECALL, METRIC_F1)}
            for category in litbank.ENTITY_CATEGORIES}


def get_phrase_counts_and_metrics(
        sentence_labels: List[List[List[str]]],
        sentence_preds: List[List[List[str]]]
) -> Tuple[dict, dict]:
    """
    Get the true positives, etc. and the precision, recall and F-scores
    for both exact and partial phrase matches.

    :param sentence_labels: Ground-truth.
    :param sentence_preds: Predictions from a model.

    :return: Counts structure is `{<category>: {<count>: (value|{<coverage>: value})}}`.
    """
    category_to_counts = _create_category_to_counts()

    # Group phrases by category.
    sentence_label_phrases = litbank.labels_to_phrases(sentence_labels)
    sentence_pred_phrases = litbank.labels_to_phrases(sentence_preds)
    for s, label_phrases_all in enumerate(sentence_label_phrases):
        category_to_label_phrases = {category: list() for category in litbank.ENTITY_CATEGORIES}
        category_to_pred_phrases = {category: list() for category in litbank.ENTITY_CATEGORIES}
        for phrase in label_phrases_all:
            category, _, _, _ = phrase
            category_to_label_phrases[category].append(phrase)
        pred_phrases_all = sentence_pred_phrases[s]
        for phrase in pred_phrases_all:
            category, _, _, _ = phrase
            category_to_pred_phrases[category].append(phrase)

        # Count true positives and denominators.
        for category in litbank.ENTITY_CATEGORIES:
            counts = category_to_counts[category]
            label_phrases = category_to_label_phrases[category]
            pred_phrases = category_to_pred_phrases[category]
            counts[COUNT_RELEVANT] += len(label_phrases)
            counts[COUNT_RETRIEVED] += len(pred_phrases)
            j = 0
            for label_phrase in label_phrases:
                while j < len(pred_phrases) and pred_phrases[j][2] <= label_phrase[1]:
                    j += 1
                if j >= len(pred_phrases):
                    break
                pred_phrase = pred_phrases[j]
                if pred_phrase[1] >= label_phrase[2]:
                    continue
                if pred_phrase[1] == label_phrase[1] and pred_phrase[2] == label_phrase[2]:
                    counts[COUNT_TRUE_POSITIVE][COVERAGE_EXACT] += 1
                t_ends = pred_phrase[2], label_phrase[2]
                t_starts = pred_phrase[1], label_phrase[1]
                counts[COUNT_TRUE_POSITIVE][COVERAGE_PARTIAL] += \
                    (min(t_ends) - max(t_starts)) / (label_phrase[2] - label_phrase[1])
                j += 1

    category_to_metrics = calculate_phrase_metrics(category_to_counts)
    return category_to_counts, category_to_metrics


def calculate_phrase_metrics(category_to_counts: dict):
    """
    Calculate the precision, recall, and F-scores given true positives, etc.

    :param category_to_counts: As returned or aggregated from `get_phrase_counts_and_metrics`.
    :return: Structure is `{<category>: {<metric>: {<coverage>: value}}}`
    """
    category_to_metrics = _create_category_to_metrics()
    for category in litbank.ENTITY_CATEGORIES:
        counts = category_to_counts[category]
        metrics = category_to_metrics[category]
        for coverage in (COVERAGE_EXACT, COVERAGE_PARTIAL):
            tp = counts[COUNT_TRUE_POSITIVE][coverage]
            retrieved = counts[COUNT_RETRIEVED]
            relevant = counts[COUNT_RELEVANT]
            precision = tp / retrieved if retrieved != 0 else 1.
            recall = tp / relevant if relevant != 0 else 1.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.
            metrics[METRIC_PRECISION][coverage] = precision
            metrics[METRIC_RECALL][coverage] = recall
            metrics[METRIC_F1][coverage] = f1

    return category_to_metrics


def calculate_fold_averages(fold_category_to_counts, fold_category_to_metrics):
    n = len(fold_category_to_metrics)
    category_to_metrics_macro = _create_category_to_metrics()
    for category_to_metrics in fold_category_to_metrics:
        for category, metrics in category_to_metrics.items():
            for metric, coverages in metrics.items():
                for coverage, value in coverages.items():
                    category_to_metrics_macro[category][metric][coverage] += value / n

    category_to_counts_micro = _create_category_to_counts()
    for category_to_counts in fold_category_to_counts:
        for category, counts in category_to_counts.items():
            counts_micro = category_to_counts_micro[category]
            for coverage, value in counts[COUNT_TRUE_POSITIVE].items():
                counts_micro[COUNT_TRUE_POSITIVE][coverage] += value
            for count in (COUNT_RETRIEVED, COUNT_RELEVANT):
                counts_micro[count] += counts[count]
    category_to_metrics_micro = calculate_phrase_metrics(category_to_counts_micro)

    return category_to_metrics_macro, category_to_metrics_micro
