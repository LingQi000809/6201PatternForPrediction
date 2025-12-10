# 6201PatternForPrediction
Repository for Fall 2025 6201 AudioContentAnalysis Final Project - Patterns for Prediction (MIREX); monophonic audio

An audio-to-audio pattern prediction system using Hidden Markov Models (HMM) to generate musical continuations from input audio.

## Overview

This project learns temporal patterns in musical sequences using an HMM trained on sliced spectrograms. 
Given an audio input, the model generates a continuation that matches the learned pattern.

**Complete Pipeline:**
```
MIDI Files (100, varying BPM)
    ↓
Tempo Normalization → 120 BPM
    ↓
FluidSynth Synthesis → 22050 Hz WAV
    ↓
Verification & Train/Test Split (60/15)
    ↓
Spectrogram Extraction and Slicing (slice for each 16th-note by default)
    ↓
K-Means Quantization (clusters of spectrogram slice)
    ↓
HMM Training
    ↓
Continuation Prediction (K-Means cluster IDs)
    ↓
Reconstruction from cluster IDs to Spectrogram 
    ↓
Audio Output
    ↓
Evaluation with Pitch/Onset extraction
```

# Dev Setup
```
virtualenv pfp
source pfp/bin/activate
pip install -r requirements.txt
brew install fluid-synth
```

## Project Structure

You need to move the PPDD datasets under a folder called "datasets" in the root of your local repository. My file structure looks like this:

```
├── datasets
│   ├── PPDD-Jul2018_aud_mono_small
│   ├── PPDD-Jul2018_aud_mono_medium
├── audio_hmm.ipynb
├── requirements.txt
├── UprightPianoKW-small-20190703.sf2
├── normalized_dataset # this is generated from audio_hmm.ipynb
├── README.md
├── ...
```

