import csv
from enum import Enum

class CsvColumns(Enum):
    ONSET = 0      # Time of the note (quarter-note beats)
    MIDI = 1          # MIDI note number
    MORPHETIC = 2  # Morph pitch number
    DUR = 3            # Duration in beats
    CHAN = 4       # MIDI channel

REST_STATE = "r"

def get_seqs_from_csv(csv_file: str, onset_offset: float):
    pitch_seq = []
    onset_seq = []

    events = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue  # skip empty lines
            # Convert all to float first
            events.append([float(cell) for cell in row])

    n = len(events)
    for i in range(n):
        onset = events[i][CsvColumns.ONSET.value] - onset_offset
        pitch = events[i][CsvColumns.MIDI.value]
        dur = events[i][CsvColumns.DUR.value]

        # append pitch
        pitch_seq.append(pitch)

        # add onset / rest logic
        onset_seq.append(onset)
        if i < n - 1:
            next_onset = events[i+1][CsvColumns.ONSET.value] - onset_offset
            gap = next_onset - (onset + dur)
            if gap > 0:
                pitch_seq.append(REST_STATE)
                onset_seq.append(onset + dur)

    pitch_seq = [str(i) for i in pitch_seq]
    onset_seq = [str(i) for i in onset_seq]
    return pitch_seq, onset_seq

def get_onset_offset(csv_file: str):
    # Onsets do not necessarily start at 0. 
    # We subtract the first onset from all subsequent onsets so that each song begins at 0.
    # This normalization ensures between-song normalization during training.
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue  # skip empty lines
            return float(row[CsvColumns.ONSET.value])
    raise ValueError(f"No valid rows found in {csv_file}")
