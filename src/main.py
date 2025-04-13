# src/main.py
import argparse
import os
import librosa

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

from .chord_recognition import chord_recognition_template
from .chord_templates import get_chord_labels
from .chroma_features import compute_chromagram_from_filename
from .evaluation import compute_eval_measures
from .io_utils import (
    annotation_to_frame_matrix,
    read_chord_annotation,
    save_chord_sequence_to_csv,
    read_beats_txt
)
from .plotting import plot_binary_time_chord, plot_chromagram, plot_eval_matrix


def main():
    parser = argparse.ArgumentParser(description="Run chord extraction pipeline.")
    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        help="Folder containing .wav and .csv files",
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, help="Folder to store results"
    )
    args = parser.parse_args()

    in_folder = args.input_folder
    out_folder = args.output_folder

    # Create output folder if needed
    os.makedirs(out_folder, exist_ok=True)

    # 1) Get all wav files
    all_wav = [f for f in os.listdir(in_folder) if f.endswith(".wav")]
    if not all_wav:
        print(f"No .wav files found in {in_folder}")
        return

    # Prepare chord label set
    chord_labels = get_chord_labels(nonchord=False)

    for wavfile in all_wav:
        base = os.path.splitext(wavfile)[0]

        csvfile_ref = base + ".csv"  # reference chord annotation
        beats_txt   = base + "(0).txt"  # reference beats annotation

        path_wav = os.path.join(in_folder, wavfile)
        path_csv = os.path.join(in_folder, csvfile_ref)
        path_beats = os.path.join(in_folder, beats_txt)

        # 2) Compute Chromagram
        X, Fs_X, x_audio, sr, x_dur = compute_chromagram_from_filename(
            path_wav, sr=22050, N=2048, H=1024, gamma=0.1, version="STFT", norm="2" # N=4096, H=2048
        )
        N_frames = X.shape[1]

        # 3) Read reference annotation (if exists), convert to matrix
        ann_segments = read_chord_annotation(path_csv)
        I_ref = np.zeros((len(chord_labels), N_frames), dtype=np.int32)
        if ann_segments:
            I_ref = annotation_to_frame_matrix(
                ann_segments, chord_labels, N_frames, Fs_X
            )

        # 4) Do chord recognition
        _, chord_max = chord_recognition_template(X, nonchord=False, norm_sim="1")
        # chord_max is (num_chords x N_frames) with 1 in the row of the recognized chord

        # 5) Evaluate
        P, R, F, TP, FP, FN = compute_eval_measures(I_ref, chord_max)

        # 6) Convert chord_max to a chord sequence (for final CSV)
        #    For each frame, find which chord is 1
        chord_idx_seq = np.argmax(chord_max, axis=0)  # shape (N_frames,)
        chord_est_seq = [chord_labels[idx] for idx in chord_idx_seq]


        ### BEATS RECOGNITION
        # 1) Cargar o leer los tiempos de beat
        beat_times = read_beats_txt(path_beats)

        # 7) Create an output subfolder (one folder per audio)
        audio_out_dir = os.path.join(out_folder, base)
        os.makedirs(audio_out_dir, exist_ok=True)

        # We only perform the 'beat sync' part if we have beats
        if beat_times:
            # Convert times (in seconds) to frames according to our sr and hop_length=512
            beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=1024)

            # Synchronize the chromagram to the beat grid
            # This groups the frames [beat_frames[i] ... beat_frames[i+1]-1] into a single column
            X_beat = librosa.util.sync(X, beat_frames, aggregate=np.mean)

            # Chord recognition with X_beat
            _, chord_max_beat = chord_recognition_template(X_beat, nonchord=False, norm_sim="1")
            # chord_max_beat will have shape (num_chords, num_beats)

            chord_idx_seq_beat = np.argmax(chord_max_beat, axis=0)  # (num_beats,)
            chord_est_seq_beat = [chord_labels[idx] for idx in chord_idx_seq_beat]

            # Save an additional CSV with one chord per beat
            # We take the beat time as 'beat_times[i]'
            csv_beat_path = os.path.join(out_folder, base, f"{base}_predicted_chords_by_beat.csv")
            with open(csv_beat_path, "w", encoding="utf-8") as fb:
                fb.write("beat_time_s,predicted_chord\n")
                for btime, c in zip(beat_times, chord_est_seq_beat):
                    fb.write(f"{btime:.3f},{c}\n")


        # 7a) Save the recognized chord sequence as CSV (time in seconds, chord)
        #     We'll say each frame is at time t = (frame_index * hop) for a hop = 2048 samples
        #     at sr=22050 => hop time = 2048/22050 ~ 0.0929 s
        hop_time = 1024.0 / sr
        times_sec = np.arange(N_frames) * hop_time
        csv_pred_path = os.path.join(audio_out_dir, f"{base}_predicted_chords.csv")
        save_chord_sequence_to_csv(times_sec, chord_est_seq, csv_pred_path)

        # 8) Generate the requested plots
        #    a) Chromagram
        plt.close("all")
        title_chroma = f"Chromagram (N={N_frames})"
        plot_chromagram(X, sr_feature=Fs_X, title=title_chroma)
        plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_chromagram.png"))
        plt.close()

        #    b) Time-chord representation of reference (if CSV found)
        if ann_segments:
            title_ref = (
                f"Time-chord representation of reference annotations (N={N_frames})"
            )
            plot_binary_time_chord(
                I_ref, chord_labels=chord_labels, sr_feature=Fs_X, title=title_ref
            )
            plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_reference.png"))
            plt.close()

        #    c) Time-chord representation of the result (the recognized chords)
        #       (We also want a CSV file with time + predicted chord, which we already saved)
        #       We'll just make a plot:
        title_est = (
            f"Time-chord representation of chord recognition result (N={N_frames})"
        )
        plot_binary_time_chord(
            chord_max, chord_labels=chord_labels, sr_feature=Fs_X, title=title_est
        )
        plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_chord_recognition.png"))
        plt.close()

        #    d) Plot the Evaluation result (with color-coded TP/FP/FN)
        #       (Only if there's a reference annotation)
        if ann_segments:
            title_eval = (
                f"Evaluation result (N={N_frames}, TP={TP}, FP={FP}, FN={FN}, "
                f"P={P:.3f}, R={R:.3f}, F={F:.3f})"
            )
            plot_eval_matrix(
                I_ref,
                chord_max,
                sr_feature=Fs_X,
                chord_labels=chord_labels,
                title=title_eval,
            )
            plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_evaluation.png"))
            plt.close()

        print(f"Done processing {wavfile}. Results in {audio_out_dir}")


if __name__ == "__main__":
    main()
