# your_project/rhythm_analysis/cli.py

import argparse
from pathlib import Path
from typing import List

from .analysis import process_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one or multiple .wav files, extract beat-based chroma/chords, and save as CSV."
    )
    parser.add_argument(
        "--audio-files",
        required=True,
        help="Comma-separated list of .wav audio file paths.",
    )
    parser.add_argument(
        "--beat-folder",
        required=True,
        help="Folder containing .txt beat annotation files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output CSV files will be saved.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for computing chromagram. Default=512.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Convert CSV list of audio files into Path objects
    audio_file_list: List[Path] = [Path(p.strip()) for p in args.audio_files.split(",")]
    beat_annotations_folder = Path(args.beat_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_filepath in audio_file_list:
        if not audio_filepath.is_file():
            print(f"Warning: audio file {audio_filepath} not found. Skipping...")
            continue

        output_csv = output_dir / f"{audio_filepath.stem}_analysis.csv"
        process_audio_file(
            audio_filepath=audio_filepath,
            beat_annotations_folder=beat_annotations_folder,
            output_csv=output_csv,
            hop_length=args.hop_length,
        )


if __name__ == "__main__":
    main()
