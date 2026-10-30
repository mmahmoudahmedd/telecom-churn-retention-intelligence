"""Retention scenario shared by training, scoring and the dashboard."""
import numpy as np

def retention(charges, probability, months=6, margin=.4, success=.2, cost=15):
    if months <= 0 or cost < 0 or not 0 <= margin <= 1 or not 0 <= success <= 1:
        raise ValueError('Invalid scenario assumptions')
    p = np.asarray(probability, dtype=float)
    c = np.asarray(charges, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Invalid probabilities or charges')
    return p * success * c * months * margin - cost
