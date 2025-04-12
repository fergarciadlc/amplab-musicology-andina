# src/plotting.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plot_chromagram(X, sr_feature=1, title="Chromagram", figsize=(20, 10)):
    """
    Simple plot of a (12 x N) chroma matrix with time on x-axis.
    sr_feature ~ frames/sec so total duration ~ N / sr_feature.
    """
    N = X.shape[1]
    dur = N / float(sr_feature)

    plt.figure(figsize=figsize)
    extent = [0, dur, 0, 12]
    plt.imshow(
        X,
        origin="lower",
        aspect="auto",
        cmap="gray_r",
        extent=extent,
        interpolation="nearest",
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Chroma bin")
    plt.tight_layout()
    # plt.show()


def plot_binary_time_chord(
    I,
    chord_labels=None,
    sr_feature=1,
    title="Time–chord representation",
    figsize=(20, 10),
):
    """
    Plots a binary matrix (num_chords x N).
    1 in a row means that chord was assigned to that frame.
    """
    if I.ndim != 2:
        raise ValueError("I must be 2D (num_chords x N).")
    num_chords, N = I.shape
    dur = N / float(sr_feature)

    plt.figure(figsize=figsize)
    plt.imshow(
        I,
        origin="lower",
        aspect="auto",
        cmap="gray_r",
        extent=[0, dur, 0, num_chords],
        interpolation="nearest",
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Chord idx")
    if chord_labels is not None:
        plt.yticks(np.arange(num_chords), chord_labels)
    plt.tight_layout()
    # plt.show()


def plot_eval_matrix(
    I_ref,
    I_est,
    sr_feature=1,
    chord_labels=None,
    title="Evaluation result",
    figsize=(20, 10),
):
    """
    Color-coded comparison:
        0=TN (white), 1=FP (red), 2=FN (pink), 3=TP (black).
    """
    if I_ref.shape != I_est.shape:
        raise ValueError("I_ref and I_est must have same shape.")
    # True positives
    TP_mask = (I_ref == 1) & (I_est == 1)
    # False positives
    FP_mask = (I_ref == 0) & (I_est == 1)
    # False negatives
    FN_mask = (I_ref == 1) & (I_est == 0)
    # Encode
    #   3 * TP + 2 * FN + 1 * FP + 0 * TN
    I_vis = (3 * TP_mask) + (2 * FN_mask) + (1 * FP_mask)  # shape = (num_chords x N)

    num_chords, N = I_ref.shape
    dur = N / float(sr_feature)

    # Colormap
    eval_cmap = colors.ListedColormap(
        [[1, 1, 1], [1, 0.3, 0.3], [1, 0.7, 0.7], [0, 0, 0]]  # TN,FP,FN,TP
    )
    eval_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    eval_norm = colors.BoundaryNorm(eval_bounds, eval_cmap.N)

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Chord")
    extent = [0, dur, 0, num_chords]

    im = plt.imshow(
        I_vis,
        origin="lower",
        aspect="auto",
        cmap=eval_cmap,
        norm=eval_norm,
        extent=extent,
    )
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["TN", "FP", "FN", "TP"])

    if chord_labels is not None:
        plt.yticks(np.arange(num_chords), chord_labels)

    plt.tight_layout()
    # plt.show()
