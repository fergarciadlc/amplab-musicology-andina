# your_project/rhythm_analysis/chords.py

from typing import Dict, Set

# Map note names to semitones (C=0, C#=1, D=2, ..., B=11)
note_to_semitone = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def build_chords_dictionary() -> Dict[str, Set[int]]:
    """
    Builds a dictionary of major/minor triad chords.
    Keys are chord names (e.g. 'C', 'Cm', 'C#', 'C#m', etc.).
    Values are sets of semitones that define that chord.

    Returns
    -------
    Dict[str, Set[int]]
        A dictionary mapping chord name -> set of semitones.
    """
    chords_dict = {}
    chroma_labels = list(note_to_semitone.keys())

    for i, note in enumerate(chroma_labels):
        maj_semitones = {(i) % 12, (i + 4) % 12, (i + 7) % 12}
        min_semitones = {(i) % 12, (i + 3) % 12, (i + 7) % 12}

        chords_dict[note] = maj_semitones
        chords_dict[note + "m"] = min_semitones

    return chords_dict


def triad_to_chord(
    triad_str: str,
    chords_dict: Dict[str, Set[int]],
    note_to_semitone_map: Dict[str, int],
) -> str:
    """
    Convert a triad string (e.g. "C-E-G") into a recognized chord name if possible.
    If not recognized, returns "Unknown".

    Parameters
    ----------
    triad_str : str
        A string such as "C-E-G".
    chords_dict : Dict[str, Set[int]]
        Dictionary of chord -> set of semitones.
    note_to_semitone_map : Dict[str, int]
        Dictionary mapping note letters to semitones.

    Returns
    -------
    str
        The recognized chord name (e.g., "C", "Cm", "G#", "Unknown", etc.).
    """
    note_names = triad_str.split("-")
    semitones = {note_to_semitone_map[note] for note in note_names}

    for chord_name, chord_semitones in chords_dict.items():
        if semitones == chord_semitones:
            return chord_name
    return "Unknown"
