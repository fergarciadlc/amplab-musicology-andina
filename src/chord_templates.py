# src/chord_templates.py
import numpy as np


def generate_chord_templates(nonchord=False):
    """
    Generate a set of chord templates for major, minor, 7th, etc.
    Returns a matrix chord_templates of shape (12, num_chords).
    If nonchord=True, adds an extra column of zeros for 'no chord'.
    """
    # Triads
    template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    # 7th chords
    template_c7 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    template_cmaj7 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    template_cm7 = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])
    template_cm7b5 = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])  # half-diminished

    # 6 chords
    template_c6 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    template_cm6 = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0])

    # dim, aug
    template_cdim = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    template_caug = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

    chord_types = [
        template_cmaj,
        template_cmin,
        template_c7,
        template_cmaj7,
        template_cm7,
        template_cm7b5,
        template_c6,
        template_cm6,
        template_cdim,
        template_caug,
    ]
    # For each chord type, build 12 versions (rolled by 0..11)
    all_templates = []
    for tmpl in chord_types:
        T = np.zeros((12, 12))
        for shift in range(12):
            T[:, shift] = np.roll(tmpl, shift)
        all_templates.append(T)
    chord_templates = np.hstack(all_templates)

    if nonchord:
        nonchord_col = np.zeros((12, 1))
        chord_templates = np.hstack([chord_templates, nonchord_col])

    return chord_templates


def get_chord_labels(nonchord=False):
    """
    Returns a list of chord labels (major, minor, 7, maj7, m7, halfdim, 6, m6, dim, aug).
    If nonchord=True, an additional 'N' is appended.
    """
    chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    labels = []

    # 1) Major triads
    for s in chroma_labels:
        labels.append(s)
    # 2) Minor triads
    for s in chroma_labels:
        labels.append(s + "m")
    # 3) Dominant 7
    for s in chroma_labels:
        labels.append(s + "7")
    # 4) Major 7
    for s in chroma_labels:
        labels.append(s + "maj7")
    # 5) Minor 7
    for s in chroma_labels:
        labels.append(s + "m7")
    # 6) Half-dim
    for s in chroma_labels:
        labels.append(s + "m7b5")
    # 7) 6
    for s in chroma_labels:
        labels.append(s + "6")
    # 8) m6
    for s in chroma_labels:
        labels.append(s + "m6")
    # 9) dim
    for s in chroma_labels:
        labels.append(s + "dim")
    # 10) aug
    for s in chroma_labels:
        labels.append(s + "aug")

    if nonchord:
        labels.append("N")
    return labels
