"""
production/utils.py
------------------
Helper functions used across multiple modules.
"""

import numpy as np

def jaccard_score(list1, list2) -> float:
    """Compute Jaccard similarity between two lists."""
    s1, s2 = set(list1), set(list2)
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

def normalize_series(series: np.ndarray, min_val=None, max_val=None) -> np.ndarray:
    """Normalize a numeric array to 0-1 scale."""
    if min_val is None:
        min_val = np.min(series)
    if max_val is None:
        max_val = np.max(series)
    denom = max(max_val - min_val, 1e-6)  # avoid div by zero
    return (series - min_val) / denom
