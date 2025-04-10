# AMPLAB - Musicology
Dataset: https://zenodo.org/records/4791394


# Harmony Analysis
## CLI Usage

Help
```bash
python -m harmony_analysis.cli -h
>>usage: cli.py [-h] --audio-files AUDIO_FILES --beat-folder BEAT_FOLDER --output-dir OUTPUT_DIR [--hop-length HOP_LENGTH]

Process one or multiple .wav files, extract beat-based chroma/chords, and save as CSV.

options:
  -h, --help            show this help message and exit
  --audio-files AUDIO_FILES
                        Comma-separated list of .wav audio file paths.
  --beat-folder BEAT_FOLDER
                        Folder containing .txt beat annotation files.
  --output-dir OUTPUT_DIR
                        Directory where output CSV files will be saved.
  --hop-length HOP_LENGTH
                        Hop length for computing chromagram. Default=512.
```

## Example
```bash
python -m harmony_analysis.cli \
    --audio-files "/Users/fernando/Downloads/4791394/rhythm_set/Audio/rh_0001.wav" \
    --beat-folder "/Users/fernando/Downloads/data/Transcriptions/rh_0001/" \
    --output-dir "output/"
```