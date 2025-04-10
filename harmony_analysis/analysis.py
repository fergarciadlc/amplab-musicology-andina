# harmony_analysis/analysis.py

from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd

from .chords import build_chords_dictionary, note_to_semitone, triad_to_chord


def load_beat_annotations(
    audio_filepath: Path,
    beat_annotations_folder: Path,
    beat_annotation_filepath: Path = None,
) -> List[float]:
    """
    Given an audio filename and a folder containing .txt beat annotations with the same name,
    loads the beat timestamps in seconds from a .txt file.

    Parameters
    ----------
    audio_filepath : Path
        Full path to the audio file (e.g., 'rh_0001.wav').
    beat_annotations_folder : Path
        Folder containing the corresponding beat annotation .txt files.

    Returns
    -------
    List[float]
        A list of beat times, in seconds.
    """
    beat_annotation_filename = audio_filepath.name.replace(".wav", ".txt")

    if beat_annotation_filepath:
        # If a specific beat annotation file is provided, use that instead
        beat_annotation_path = beat_annotation_filepath
    else:
        beat_annotation_path = beat_annotations_folder / beat_annotation_filename

    beats_times = []
    with open(beat_annotation_path, "r") as f:
        for line in f:
            beats_times.append(float(line.strip()))

    return beats_times


def compute_beat_chroma(
    audio_data: np.ndarray,
    sr: int,
    beats_timestamps: List[float],
    hop_length: int = 512,
) -> pd.DataFrame:
    """
    Computes the average chroma between consecutive beat timestamps.

    Parameters
    ----------
    audio_data : np.ndarray
        The raw audio samples (mono).
    sr : int
        Sampling rate of the audio.
    beats_timestamps : List[float]
        List of beat times in seconds.
    hop_length : int
        The hop length used to compute the chromagram.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        [beat_start_time, beat_end_time, chroma_0, ..., chroma_11].
    """

    # Compute chroma using CQT (can also use chroma_stft or chroma_cens)
    chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sr, hop_length=hop_length)

    # Convert beat times to frame indices
    beat_frames = librosa.time_to_frames(beats_timestamps, sr=sr, hop_length=hop_length)

    # Gather results
    results = []
    for i in range(len(beat_frames) - 1):
        start_frame = beat_frames[i]
        end_frame = beat_frames[i + 1]

        # Clamp to chromagram shape
        end_frame = min(end_frame, chromagram.shape[1])

        segment = chromagram[:, start_frame:end_frame]
        mean_chroma = np.mean(segment, axis=1)

        results.append(
            {
                "beat_start_time": beats_timestamps[i],
                "beat_end_time": beats_timestamps[i + 1],
                **{f"chroma_{c}": mean_chroma[c] for c in range(12)},
            }
        )

    df = pd.DataFrame(results)
    return df


def add_triads_and_chords(beat_chroma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'chroma_0' through 'chroma_11', compute:
      - The top 3 notes (as a triad string)
      - The chord name, if recognized (e.g. C, Am, D#, etc.)

    Parameters
    ----------
    beat_chroma_df : pd.DataFrame
        DataFrame containing columns "chroma_0" ... "chroma_11".

    Returns
    -------
    pd.DataFrame
        The same DataFrame with two new columns: 'triad' and 'chord'.
    """
    # Mapping from chroma_i to note names
    chroma_notes_mapping = {
        "chroma_0": "C",
        "chroma_1": "C#",
        "chroma_2": "D",
        "chroma_3": "D#",
        "chroma_4": "E",
        "chroma_5": "F",
        "chroma_6": "F#",
        "chroma_7": "G",
        "chroma_8": "G#",
        "chroma_9": "A",
        "chroma_10": "A#",
        "chroma_11": "B",
    }
    chroma_cols = [f"chroma_{i}" for i in range(12)]

    def get_top_n_notes(row: pd.Series, n=3) -> str:
        top_chroma_cols = row.nlargest(
            n
        ).index  # e.g. ["chroma_0", "chroma_4", "chroma_7"]
        notes = [chroma_notes_mapping[col] for col in top_chroma_cols]
        return "-".join(notes)  # e.g. "C-E-G"

    beat_chroma_df["triad"] = beat_chroma_df[chroma_cols].apply(
        get_top_n_notes, n=3, axis=1
    )

    # Build chord templates once
    chords_dict = build_chords_dictionary()

    # Map each triad to a recognized chord (or "Unknown")
    beat_chroma_df["chord"] = beat_chroma_df["triad"].apply(
        lambda triad_str: triad_to_chord(triad_str, chords_dict, note_to_semitone)
    )

    beat_chroma_df["tetrad"] = beat_chroma_df[chroma_cols].apply(
        get_top_n_notes, n=4, axis=1
    )

    return beat_chroma_df


def process_audio_file(
    audio_filepath: Path,
    beat_annotations_folder: Path,
    output_csv: Path,
    hop_length: int = 512,
    beat_annotation_filepath: Path = None,
) -> None:
    """
    High-level function to load one audio file, compute the beat-level chroma,
    triads, and chord names, then save the final results to CSV.

    Parameters
    ----------
    audio_filepath : Path
        Full path to the .wav audio file.
    beat_annotations_folder : Path
        Path to the folder containing .txt beat annotations.
    output_csv : Path
        Path where the final CSV file will be saved.
    hop_length : int
        Hop length for computing chromagram.
    """
    # 1) Load audio
    x, sr = librosa.load(audio_filepath, sr=None, mono=True)

    # 2) Load beats
    beats_timestamps = load_beat_annotations(
        audio_filepath, beat_annotations_folder, beat_annotation_filepath
    )

    # Edge case: if less than 2 beats, we can't do intervals
    if len(beats_timestamps) < 2:
        print(f"Warning: Not enough beats found in {audio_filepath.name}, skipping.")
        return

    # 3) Compute beat-level chroma
    beat_chroma_df = compute_beat_chroma(x, sr, beats_timestamps, hop_length=hop_length)

    # 4) Compute triads + chord labels
    beat_chroma_df = add_triads_and_chords(beat_chroma_df)

    # 5) Save final DataFrame to CSV
    beat_chroma_df.to_csv(output_csv, index=False)
    print(f"Saved analysis to {output_csv}")
