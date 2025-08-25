# 🎵 MusiPy

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-orange)](https://streamlit.io/)

> **MusiPy** — an end-to-end music generation and analytics platform built with Python, TensorFlow, and Streamlit. Train LSTM models on MIDI data, generate melodies, visualize musical patterns, and download MIDI compositions.

---

## Features

- **Quick Music Generation**
  - Instant generation using genre-specific pattern banks (Classical, Jazz, Ambient, Folk, Electronic).
  - Adjustable tempo, number of notes, and creativity (temperature).
  - Preview and download generated MIDI files.

- **Custom Model Training**
  - Upload MIDI files or use built-in sample datasets.
  - Sequence preparation and tokenization of notes and chords.
  - LSTM-based model with configurable epochs and batch size.
  - Progress tracking, early stopping, and best-model checkpointing.

- **Music Generation with Trained Models**
  - Seed-based generation using trained LSTM model.
  - Temperature-based sampling for controllable creativity.
  - Converts predicted token sequences back to note/chord strings and MIDI.

- **Analytics Dashboard**
  - Training & validation loss/accuracy plots.
  - Note frequency distribution and sequence visualization.
  - Model architecture summary and parameter counts.

- **MIDI Utilities**
  - Robust MIDI note & chord extraction using `music21`.
  - Conversion of note/chord sequences to `.mid` files for playback and DAWs.

- **Polished UI**
  - Streamlit frontend with custom CSS, interactive plots (Plotly), and downloadable MIDI files.

---

## Demo

Run the app locally and open `http://localhost:8501` after starting Streamlit. The app provides three modes:
1. **Quick Generate** — instant melodies from templates.
2. **Train Custom Model** — upload MIDI, preprocess, train, and save models.
3. **Model Analytics** — visualize training history and model internals.

---

## Tech Stack

- **Languages:** Python 3.9+
- **Web UI:** Streamlit
- **ML / DL:** TensorFlow, Keras (LSTM)
- **Music processing:** music21, pretty_midi
- **Data / Viz:** NumPy, Pandas, Plotly
- **Utilities:** tempfile, pickle, zipfile

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shikhar4904/MusiPy.git
cd MusiPy
```

2. Create a virtual environment (recommended) and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run main.py
```

---

## Usage

- Use **Quick Generate** to explore the styles and download sample MIDI files.
- Upload MIDI files in **Train Custom Model** to generate personalized models.
- Use **Generate** tab to create music from the trained model and download `.mid`.
- Inspect **Analytics** tab to tune training hyperparameters and see model performance.

---

## Model Architecture

- Three stacked LSTM layers with 512 units each, dropout/recurrent dropout for regularization.
- Fully connected Dense layers culminating in softmax over the vocabulary.
- Categorical crossentropy loss, Adam optimizer.

---

## Data Processing

- Extract notes and chords from MIDI using `music21`.
- Tokenize unique note/chord strings into integer indices.
- Create sliding-window sequences (sequence_length = 100 by default).
- Normalize inputs and one-hot encode outputs for classification-style next-token prediction.

---

## Training

- Uses `ModelCheckpoint` (save best by `val_loss`) and `EarlyStopping`.
- Typical training loop:
  - Split data into training/validation sets.
  - Train with configurable `epochs` and `batch_size`.
  - Monitor `loss`, `val_loss`, `accuracy`, and `val_accuracy`.

---

## Generation & MIDI Conversion

- Seed with a sequence of token indices and sample next tokens using temperature scaling.
- Convert indices back to note/chord strings and write a `.mid` file via music21's stream API.

---

## Analytics & Visualization

- Interactive plots (Plotly) for:
  - Training/validation loss & accuracy through epochs.
  - Top-n most used notes.
  - Note sequence timeline (first N notes).

---

## Future Enhancements

- Replace LSTM with Transformer architectures (e.g., Music Transformer) for improved long-term structure.
- Add multi-instrument orchestration and instrument assignment.
- Real-time audio playback & DAW export (MIDI + stems).
- Web deployment (Heroku / Render / Streamlit Cloud / Docker).

---

Happy composing! 🎶
