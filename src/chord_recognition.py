# src/chord_recognition.py
import numpy as np

from .chord_templates import generate_chord_templates
from .chroma_features import normalize_feature_sequence


def chord_recognition_template(X, nonchord=False, norm_sim="1"):
    """
    Conduct template-based chord recognition.
    X is a (12 x N) chromagram.

    Returns:
        chord_sim: (num_chords x N) similarity matrix
        chord_max: (num_chords x N) binarized matrix with 1 for max-sim chord in each frame
    """
    chord_templates = generate_chord_templates(nonchord=nonchord)
    # Normalize columns of X and chord templates to ensure fair comparison
    X_norm = normalize_feature_sequence(X, norm="2")
    T_norm = normalize_feature_sequence(chord_templates, norm="2")

    # Similarity (dot product)
    chord_sim = np.matmul(T_norm.T, X_norm)

    # Optionally normalize similarity columns
    if norm_sim is not None:
        chord_sim = normalize_feature_sequence(chord_sim, norm=norm_sim)

    # Pick max chord in each frame
    chord_max = np.zeros_like(chord_sim)
    max_idx = np.argmax(chord_sim, axis=0)
    for n, idx in enumerate(max_idx):
        chord_max[idx, n] = 1

    return chord_sim, chord_max
