# src/io_utils.py
import csv
import os

import numpy as np

# def read_chord_annotation(csv_path):
#     """
#     Reads a CSV with columns like: start_time_seconds, end_time_seconds, chord_label
#     Returns a list of segments: [(start, end, chord_str), ...]
#     or an empty list if not found or invalid.
#     """
#     segments = []
#     if not os.path.isfile(csv_path):
#         return segments
#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             try:
#                 start = float(row[0])
#                 end = float(row[1])
#                 label = row[2].strip()
#                 segments.append((start, end, label))
#             except:
#                 pass
#     return segments


def read_chord_annotation(csv_path):
    """
    Reads a CSV with columns like: start_time_seconds, end_time_seconds, chord_label
    Returns a list of segments: [(start, end, chord_str), ...]
    """
    import csv
    import os

    segments = []
    if not os.path.isfile(csv_path):
        return segments

    with open(csv_path, "r", encoding="utf-8") as f:
        # Use semicolon delimiter and double-quote as quotechar:
        reader = csv.reader(f, delimiter=";", quotechar='"')

        # Optionally skip the very first row if it looks like a header
        # e.g., if row[0] == "Start" or something
        first_line = True
        for row in reader:
            # If the first line looks like a header, skip it
            if first_line and row and (row[0].strip().lower() in ["start", '"start"']):
                first_line = False
                continue
            first_line = False

            if len(row) < 3:
                continue

            try:
                start = float(row[0].strip())
                end = float(row[1].strip())
                label = row[2].strip()
                segments.append((start, end, label))
            except ValueError:
                # If float conversion fails, skip this row
                continue

    return segments


def annotation_to_frame_matrix(segments, chord_labels, N, sr_feature):
    """
    Convert segment-based annotation [(start_sec, end_sec, chord_label), ...]
    into a binary matrix (num_chords x N).
    We assume 1 chord per time; if unknown chord, we do nothing (all 0).
    """
    num_chords = len(chord_labels)
    I_ref = np.zeros((num_chords, N), dtype=np.int32)

    for start_sec, end_sec, label in segments:
        start_idx = int(round(start_sec * sr_feature))
        end_idx = int(round(end_sec * sr_feature))
        # clamp
        start_idx = max(0, start_idx)
        end_idx = min(N, end_idx)
        if label in chord_labels:
            row = chord_labels.index(label)
            I_ref[row, start_idx:end_idx] = 1
    return I_ref


def save_chord_sequence_to_csv(times_s, chord_seq, out_csv):
    """
    Save recognized chord sequence as CSV with columns: time (sec), predicted_chord.
    We assume times_s has length N, chord_seq has length N, etc.
    Typically, times_s = np.arange(N) * (hopsize_s), or something similar.
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "predicted_chord"])
        for t, c in zip(times_s, chord_seq):
            writer.writerow([f"{t:.3f}", c])


def read_beats_txt(filename):
    """
    Lee un archivo .txt donde cada línea contiene un tiempo de beat en segundos.
    Retorna una lista de floats.
    """
    beat_times = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # si no está vacío
                try:
                    t = float(line)
                    beat_times.append(t)
                except ValueError:
                    pass  # ignora líneas que no puedan convertirse a float
    return beat_times