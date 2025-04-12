
import argparse
import os

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

    print(f"[INFO] Input folder: {in_folder}")
    print(f"[INFO] Output folder: {out_folder}")

    os.makedirs(out_folder, exist_ok=True)

    all_wav = [f for f in os.listdir(in_folder) if f.endswith(".wav")]
    print(f"[INFO] Found {len(all_wav)} .wav files: {all_wav}")
    if not all_wav:
        print(f"[WARNING] No .wav files found in {in_folder}")
        return

    chord_labels = get_chord_labels(nonchord=False)

    for wavfile in all_wav:
        print(f"[INFO] Processing {wavfile}")
        base = os.path.splitext(wavfile)[0]
        csvfile_ref = base + ".csv"
        path_wav = os.path.join(in_folder, wavfile)
        path_csv = os.path.join(in_folder, csvfile_ref)

        print(f"[INFO] Loading audio: {path_wav}")
        X, Fs_X, x_audio, sr, x_dur = compute_chromagram_from_filename(
            path_wav, sr=22050, N=4096, H=2048, gamma=0.1, version="CQT", norm="2"
        )
        N_frames = X.shape[1]
        print(f"[INFO] Chromagram shape: {X.shape}, Fs_X: {Fs_X:.2f}, Duration: {x_dur:.2f} s")

        ann_segments = read_chord_annotation(path_csv)
        I_ref = np.zeros((len(chord_labels), N_frames), dtype=np.int32)
        if ann_segments:
            print(f"[INFO] Reference annotations found in {csvfile_ref}")
            I_ref = annotation_to_frame_matrix(
                ann_segments, chord_labels, N_frames, Fs_X
            )
        else:
            print(f"[WARNING] No annotation file found: {csvfile_ref}")

        print("[INFO] Performing chord recognition...")
        _, chord_max = chord_recognition_template(X, nonchord=False, norm_sim="1")

        print("[INFO] Evaluating results...")
        P, R, F, TP, FP, FN = compute_eval_measures(I_ref, chord_max)

        chord_idx_seq = np.argmax(chord_max, axis=0)
        chord_est_seq = [chord_labels[idx] for idx in chord_idx_seq]

        audio_out_dir = os.path.join(out_folder, base)
        os.makedirs(audio_out_dir, exist_ok=True)

        hop_time = 2048.0 / sr
        times_sec = np.arange(N_frames) * hop_time
        csv_pred_path = os.path.join(audio_out_dir, f"{base}_predicted_chords.csv")
        save_chord_sequence_to_csv(times_sec, chord_est_seq, csv_pred_path)
        print(f"[INFO] Saved predicted chord sequence to {csv_pred_path}")

        # Plots
        print("[INFO] Generating plots...")
        plt.close("all")
        plot_chromagram(X, sr_feature=Fs_X, title=f"Chromagram (N={N_frames})")
        plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_chromagram.png"))
        plt.close()

        if ann_segments:
            plot_binary_time_chord(
                I_ref, chord_labels=chord_labels, sr_feature=Fs_X,
                title=f"Time-chord representation of reference (N={N_frames})"
            )
            plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_reference.png"))
            plt.close()

        plot_binary_time_chord(
            chord_max, chord_labels=chord_labels, sr_feature=Fs_X,
            title=f"Chord recognition result (N={N_frames})"
        )
        plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_chord_recognition.png"))
        plt.close()

        if ann_segments:
            plot_eval_matrix(
                I_ref, chord_max, sr_feature=Fs_X, chord_labels=chord_labels,
                title=f"Eval: N={N_frames}, TP={TP}, FP={FP}, FN={FN}, P={P:.3f}, R={R:.3f}, F={F:.3f}"
            )
            plt.savefig(os.path.join(audio_out_dir, f"{base}_plot_evaluation.png"))
            plt.close()

        print(f"[INFO] Done processing {wavfile}. Results saved to {audio_out_dir}")


if __name__ == "__main__":
    main()
