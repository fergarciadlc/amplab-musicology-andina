# src/evaluation.py
import numpy as np


def compute_eval_measures(I_ref, I_est):
    """
    Compare reference annotation (I_ref) vs. estimated (I_est), each a binary matrix (num_chords x N).
    Returns: (Precision, Recall, F-measure, TP, FP, FN).
    """
    assert I_ref.shape == I_est.shape, "Reference and estimate must have same shape"

    TP = np.sum(np.logical_and(I_ref, I_est))
    FP = np.sum(I_est) - TP
    FN = np.sum(I_ref) - TP

    P, R, F = 0, 0, 0
    if TP > 0:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)

    return P, R, F, TP, FP, FN
