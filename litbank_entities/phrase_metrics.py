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


def _create_category_counts(k):
    return [{
        COUNT_TRUE_POSITIVE: {
            COVERAGE_EXACT: 0,
            COVERAGE_PARTIAL: 0,
        },
        COUNT_RETRIEVED: 0,
        COUNT_RELEVANT: 0
    } for _ in range(k)]


def _create_category_metrics(k):
    return [{name: {coverage: 0
                    for coverage in (COVERAGE_EXACT, COVERAGE_PARTIAL)}
             for name in (METRIC_PRECISION, METRIC_RECALL, METRIC_F1)}
            for _ in range(k)]


def get_phrase_counts_and_metrics(
        sentence_labels: List[List[List[str]]],
        sentence_preds: List[List[List[str]]],
        categories: List[str]
) -> Tuple[List[dict], List[dict]]:
    """
    Get the true positives, etc. and the precision, recall and F-scores
    for both exact and partial phrase matches.

    :param sentence_labels: Ground-truth.
    :param sentence_preds: Predictions from a model.
    :param categories: Iterable of category names.

    :return: Counts structure is `[{<count>: (value|{<coverage>: value})}]` for each category.
    """
    category_counts = _create_category_counts(len(categories))

    # Group phrases by category.
    category_sentence_label_phrases = litbank.get_category_sentence_phrases(sentence_labels, categories)
    category_sentence_pred_phrases = litbank.get_category_sentence_phrases(sentence_preds, categories)
    for category_id, sentence_label_phrases in enumerate(category_sentence_label_phrases):
        sentence_pred_phrases = category_sentence_pred_phrases[category_id]
        for i, label_phrases in enumerate(sentence_label_phrases):
            pred_phrases = sentence_pred_phrases[i]

            # Count true positives and denominators.
            counts = category_counts[category_id]
            counts[COUNT_RELEVANT] += len(label_phrases)
            counts[COUNT_RETRIEVED] += len(pred_phrases)
            j = 0
            for label_phrase in label_phrases:
                while j < len(pred_phrases) and pred_phrases[j][1] <= label_phrase[0]:
                    j += 1
                if j >= len(pred_phrases):
                    break
                pred_phrase = pred_phrases[j]
                if pred_phrase[0] >= label_phrase[1]:
                    continue
                if pred_phrase[0] == label_phrase[0] and pred_phrase[1] == label_phrase[1]:
                    counts[COUNT_TRUE_POSITIVE][COVERAGE_EXACT] += 1
                t_ends = pred_phrase[1], label_phrase[1]
                t_starts = pred_phrase[0], label_phrase[0]
                counts[COUNT_TRUE_POSITIVE][COVERAGE_PARTIAL] += \
                    (min(t_ends) - max(t_starts)) / (label_phrase[1] - label_phrase[0])
                j += 1

    category_metrics = calculate_phrase_metrics(category_counts)
    return category_counts, category_metrics


def calculate_phrase_metrics(category_counts: List[dict]) -> List[dict]:
    """
    Calculate the precision, recall, and F-scores given true positives, etc.

    :param category_counts: As returned or aggregated from `get_phrase_counts_and_metrics`.
    :return: Structure is `{<category>: {<metric>: {<coverage>: value}}}`
    """
    category_metrics = _create_category_metrics(len(category_counts))
    for category_id, counts in enumerate(category_counts):
        metrics = category_metrics[category_id]
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

    return category_metrics


def calculate_fold_averages(fold_category_counts, fold_category_metrics):
    n = len(fold_category_metrics)
    k = len(fold_category_metrics[0])
    category_metrics_macro = _create_category_metrics(k)
    for category_metrics in fold_category_metrics:
        for category_id, metrics in enumerate(category_metrics):
            for metric, coverages in metrics.items():
                for coverage, value in coverages.items():
                    category_metrics_macro[category_id][metric][coverage] += value / n

    category_counts_micro = _create_category_counts(k)
    for category_counts in fold_category_counts:
        for category_id, counts in enumerate(category_counts):
            counts_micro = category_counts_micro[category_id]
            for coverage, value in counts[COUNT_TRUE_POSITIVE].items():
                counts_micro[COUNT_TRUE_POSITIVE][coverage] += value
            for count in (COUNT_RETRIEVED, COUNT_RELEVANT):
                counts_micro[count] += counts[count]
    category_metrics_micro = calculate_phrase_metrics(category_counts_micro)

    return category_metrics_macro, category_metrics_micro
