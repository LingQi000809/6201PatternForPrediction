import csv
from mido import MidiFile, MidiTrack, Message
from preprocessing import CsvColumns

def sequence_to_midi(
        midi_file_path, 
        onset_offset, pitch_seq, onset_seq, 
        csv_file_path = None,
        velocity=64):
    """
    Convert pitch + onset sequences into a MIDI file and optionally export to CSV.
    
    Args:
        onset_offset: first onset to normalize sequence
        pitch_seq: list of pitches (with 'r' for rests)
        onset_seq: list of onsets in quarter-note beats
        midi_file_path: path to save MIDI
        csv_file_path: optional path to save CSV
        velocity: MIDI note velocity
    """
    assert len(pitch_seq) == len(onset_seq), "pitch_seq and onset_seq must be same length"
    pitch_seq = [float(i) if i != "r" else -1 for i in pitch_seq]
    onset_seq = [float(i) for i in onset_seq]

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    csv_rows = []
    tick_from_prev_note = 0
    for i, pitch in enumerate(pitch_seq):
        onset = onset_seq[i] + onset_offset

        # compute note duration as time until next onset
        if i+1 >= len(onset_seq):
            # last note → set a default short duration
            next_onset = onset + 0.25  # quarter-beat default
        else:
            next_onset = onset_seq[i+1] + onset_offset
        note_duration_beats = next_onset - onset
        note_duration_ticks = int(note_duration_beats * mid.ticks_per_beat)

        if pitch != -1:
            # the time for note_on and note_off events is always the delta tick from the pervious event
            track.append(Message('note_on', note=int(float(pitch)), velocity=velocity, time=tick_from_prev_note))
            track.append(Message('note_off', note=int(float(pitch)), velocity=velocity, time=note_duration_ticks))

            tick_from_prev_note = 0

            # Add row for CSV
            if csv_file_path:
                csv_rows.append([onset, pitch, note_duration_beats])
        else:
            # rest → no note, just advance prev_onset
            tick_from_prev_note = note_duration_ticks

    mid.save(midi_file_path)
    print(f"MIDI saved to {midi_file_path}")

    # Save CSV if path provided
    if csv_file_path:
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"CSV saved to {csv_file_path}")
    return midi_file_path


import pretty_midi
import soundfile as sf
def midi_to_wav(midi_file, wav_file, fs=44100, soundfont=None):
    """
    Render a MIDI file to WAV using pretty_midi's built-in synthesizer.

    Parameters:
        midi_file (str): Path to the MIDI file.
        wav_file (str): Path to output WAV file.
        fs (int): Sampling rate (default 44100 Hz).
        soundfont (str): Optional SoundFont path (pretty_midi can use FluidSynth if provided).

    Returns:
        str: Path to the saved WAV file.
    """
    # Load MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Synthesize audio (numpy array)
    audio = midi_data.fluidsynth(fs=fs, sf2_path=soundfont) if soundfont else midi_data.synthesize(fs=fs)

    # Save to WAV
    sf.write(wav_file, audio, fs)
    print(f"WAV saved to {wav_file}")
    return wav_file