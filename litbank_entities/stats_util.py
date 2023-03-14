import operator
from collections import defaultdict


def get_item_counts(items, key=None, reverse=True):
    if key is None:
        key = operator.itemgetter(1, 0)
    item_to_count = defaultdict(int)
    for item in items:
        item_to_count[item] += 1
    return sorted(item_to_count.items(), key=key, reverse=reverse)
