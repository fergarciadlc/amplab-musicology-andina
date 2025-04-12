# src/chroma_features.py
import librosa
import numpy as np


def normalize_feature_sequence(X, norm="2", threshold=1e-4, v=None):
    """
    Normalize columns of feature matrix X (shape: K x N).
    Supported norms: '1', '2', 'max', 'z'.
    """
    assert norm in ["1", "2", "max", "z"], "Norm must be one of '1','2','max','z'"
    K, N = X.shape
    X_norm = np.zeros((K, N))

    for n in range(N):
        col = X[:, n]
        if norm == "1":
            val = np.sum(np.abs(col))
            if val > threshold:
                X_norm[:, n] = col / val
            else:
                X_norm[:, n] = v if v is not None else np.ones(K) / K
        elif norm == "2":
            val = np.sqrt(np.sum(col**2))
            if val > threshold:
                X_norm[:, n] = col / val
            else:
                X_norm[:, n] = v if v is not None else np.ones(K) / np.sqrt(K)
        elif norm == "max":
            val = np.max(np.abs(col))
            if val > threshold:
                X_norm[:, n] = col / val
            else:
                X_norm[:, n] = v if v is not None else np.ones(K)
        elif norm == "z":
            mu = np.mean(col)
            sigma = np.std(col, ddof=1)
            if sigma > threshold:
                X_norm[:, n] = (col - mu) / sigma
            else:
                X_norm[:, n] = v if v is not None else np.zeros(K)
    return X_norm


def compute_chromagram_from_filename(
    fn_wav, sr=22050, N=4096, H=2048, gamma=None, version="STFT", norm="2"
):
    """
    Load WAV/Audio file (fn_wav) and compute a chromagram.

    version in ['STFT','CQT','IIR'].

    Returns:
        X     : (12 x num_frames) chroma matrix
        Fs_X  : feature rate (e.g. sr / H)
        x     : loaded audio
        sr    : actual sampling rate
        x_dur : duration in seconds
    """
    # 1) Load audio
    x, sr = librosa.load(fn_wav, sr=sr)
    x_dur = len(x) / sr

    # 2) Compute chroma
    if version == "STFT":
        X_stft = librosa.stft(x, n_fft=N, hop_length=H)
        X_pow = np.abs(X_stft) ** 2
        if gamma is not None:
            X_pow = np.log(1 + gamma * X_pow)
        X = librosa.feature.chroma_stft(
            S=X_pow, sr=sr, hop_length=H, n_fft=N, norm=None
        )
    elif version == "CQT":
        X = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=H, norm=None)
    elif version == "IIR":
        X_iir = librosa.iirt(x, sr=sr, win_length=N, hop_length=H)
        if gamma is not None:
            X_iir = np.log(1 + gamma * X_iir)
        # Then use cqt-based chroma
        X = librosa.feature.chroma_cqt(C=X_iir, sr=sr, norm=None)
    else:
        raise ValueError("version must be 'STFT','CQT','IIR'")

    # 3) Normalize if requested
    if norm is not None:
        X = normalize_feature_sequence(X, norm=norm)

    # 4) Feature rate
    Fs_X = sr / float(H)

    return X, Fs_X, x, sr, x_dur
