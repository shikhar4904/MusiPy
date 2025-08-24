# AI Music Composer - Complete Application
# This is a fully functional AI music generation system with Streamlit frontend

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import music21
from music21 import converter, instrument, note, chord, stream, duration, pitch, meter, key
import pretty_midi
import io
import pickle
import os
import tempfile
import base64
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import zipfile
import requests
from pathlib import Path
import threading
import time

# Configure Streamlit
st.set_page_config(
    page_title="AI Music Composer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class MusicDataProcessor:
    """Handles MIDI data processing and tokenization"""
    
    def __init__(self):
        self.notes = []
        self.note_to_int = {}
        self.int_to_note = {}
        self.sequence_length = 100
        
    def extract_notes_from_midi(self, midi_file_path: str) -> List[str]:
        """Extract notes from MIDI file"""
        try:
            midi = converter.parse(midi_file_path)
            notes_to_parse = None
            
            # Get instrument parts
            parts = instrument.partitionByInstrument(midi)
            
            if parts:  # Multi-instrument MIDI
                notes_to_parse = parts.parts[0].recurse()
            else:  # Single instrument MIDI
                notes_to_parse = midi.flat.notes
                
            extracted_notes = []
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    extracted_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    extracted_notes.append('.'.join(str(n) for n in element.normalOrder))
                    
            return extracted_notes
            
        except Exception as e:
            st.error(f"Error processing MIDI file: {str(e)}")
            return []
    
    def create_sample_dataset(self) -> List[str]:
        """Create a sample dataset of musical notes for demonstration"""
        # Classical music patterns in C major
        notes = []
        
        # Scale patterns
        c_major_scale = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        
        # Generate various patterns
        for _ in range(50):
            # Ascending scale
            notes.extend(c_major_scale)
            # Descending scale
            notes.extend(c_major_scale[::-1])
            
            # Arpeggios
            arpeggios = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4']
            notes.extend(arpeggios * 2)
            
            # Common chord progressions
            chord_prog = ['C4', 'F4', 'G4', 'C4'] * 4
            notes.extend(chord_prog)
            
            # Melodic patterns
            melody = ['C4', 'D4', 'E4', 'D4', 'C4', 'G4', 'F4', 'E4', 'D4', 'C4']
            notes.extend(melody * 3)
            
        return notes
    
    def prepare_sequences(self, notes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        # Create mappings
        unique_notes = sorted(set(notes))
        self.note_to_int = {note: idx for idx, note in enumerate(unique_notes)}
        self.int_to_note = {idx: note for idx, note in enumerate(unique_notes)}
        
        # Create sequences
        network_input = []
        network_output = []
        
        for i in range(0, len(notes) - self.sequence_length):
            sequence_in = notes[i:i + self.sequence_length]
            sequence_out = notes[i + self.sequence_length]
            
            network_input.append([self.note_to_int[note] for note in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
            
        n_patterns = len(network_input)
        
        # Reshape for LSTM
        network_input = np.reshape(network_input, (n_patterns, self.sequence_length, 1))
        network_input = network_input / float(len(unique_notes))
        
        # One-hot encode output
        network_output = tf.keras.utils.to_categorical(network_output)
        
        return network_input, network_output

class MusicGenerator:
    """LSTM-based music generation model"""
    
    def __init__(self, n_vocab: int):
        self.n_vocab = n_vocab
        self.model = None
        
    def build_model(self) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(512, input_shape=(100, 1), return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            LSTM(512, dropout=0.3, recurrent_dropout=0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.n_vocab, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64) -> Dict:
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_music_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def generate_music(self, seed_sequence: List[int], num_notes: int = 100, 
                      temperature: float = 1.0) -> List[int]:
        """Generate music using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        generated = []
        pattern = seed_sequence.copy()
        
        for _ in range(num_notes):
            # Prepare input
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            
            # Predict next note
            prediction = self.model.predict(x, verbose=0)[0]
            
            # Apply temperature
            prediction = np.log(prediction + 1e-8) / temperature
            prediction = np.exp(prediction) / np.sum(np.exp(prediction))
            
            # Sample from distribution
            index = np.random.choice(len(prediction), p=prediction)
            
            generated.append(index)
            pattern.append(index)
            pattern = pattern[1:]  # Remove first element
            
        return generated

class MIDIConverter:
    """Convert generated sequences to MIDI"""
    
    @staticmethod
    def create_midi_from_notes(notes: List[str], filename: str = "generated_music.mid"):
        """Create MIDI file from note sequence"""
        offset = 0
        output_notes = []
        
        for pattern in notes:
            # Check if it's a chord
            if '.' in pattern:
                notes_in_chord = pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            
            offset += 0.5
        
        # Create stream
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=filename)
        return filename

# Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 AI Music Composer</h1>', unsafe_allow_html=True)
    st.markdown("Generate beautiful melodies using artificial intelligence!")
    
    # Sidebar
    st.sidebar.markdown('<div class="section-header">🎛️ Controls</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.processor = None
        st.session_state.trained = False
        st.session_state.training_history = None
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🎹 Quick Generate", "🔧 Train Custom Model", "📊 Model Analytics"]
    )
    
    if mode == "🎹 Quick Generate":
        quick_generation_mode()
    elif mode == "🔧 Train Custom Model":
        training_mode()
    elif mode == "📊 Model Analytics":
        analytics_mode()

def quick_generation_mode():
    """Quick music generation with pre-trained patterns"""
    st.markdown('<div class="section-header">🎹 Quick Music Generation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Generate music instantly using built-in patterns!")
        
        # Generation parameters
        genre = st.selectbox(
            "🎼 Select Genre/Style",
            ["Classical", "Jazz", "Ambient", "Folk", "Electronic"]
        )
        
        tempo = st.slider("🥁 Tempo (BPM)", 60, 180, 120)
        num_notes = st.slider("📝 Number of Notes", 20, 200, 100)
        temperature = st.slider("🌡️ Creativity (Temperature)", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        st.markdown("### 🎵 Generation Preview")
        if st.button("🎲 Generate Music", key="quick_gen"):
            generate_quick_music(genre, tempo, num_notes, temperature)

def generate_quick_music(genre: str, tempo: int, num_notes: int, temperature: float):
    """Generate music with predefined patterns"""
    with st.spinner("🎵 Composing your melody..."):
        # Create processor
        processor = MusicDataProcessor()
        
        # Generate genre-specific patterns
        if genre == "Classical":
            base_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'] * 20
        elif genre == "Jazz":
            base_notes = ['C4', 'E4', 'G4', 'Bb4', 'D5', 'F4', 'A4', 'C5'] * 15
        elif genre == "Ambient":
            base_notes = ['C4', 'G4', 'C5', 'E5', 'G5', 'E5', 'C5', 'G4'] * 18
        elif genre == "Folk":
            base_notes = ['G4', 'A4', 'B4', 'C5', 'D5', 'C5', 'B4', 'A4', 'G4'] * 16
        else:  # Electronic
            base_notes = ['C4', 'Eb4', 'F4', 'G4', 'Bb4', 'C5', 'Bb4', 'G4'] * 17
        
        # Add some randomization
        import random
        notes_pool = base_notes.copy()
        for _ in range(num_notes // 4):
            # Add some variations
            random_note = random.choice(['D4', 'E4', 'F4', 'A4', 'B4'])
            notes_pool.append(random_note)
        
        # Shuffle for variation
        random.shuffle(notes_pool)
        generated_notes = notes_pool[:num_notes]
        
        # Create MIDI
        midi_file = MIDIConverter.create_midi_from_notes(generated_notes, "quick_generated.mid")
        
        # Display results
        st.success("🎉 Music generated successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📝 Notes Generated", len(generated_notes))
        
        with col2:
            st.metric("🎼 Unique Notes", len(set(generated_notes)))
        
        with col3:
            st.metric("⏱️ Estimated Duration", f"{len(generated_notes) * 0.5:.1f}s")
        
        # Download button
        with open(midi_file, "rb") as f:
            st.download_button(
                label="📥 Download MIDI",
                data=f.read(),
                file_name=f"{genre.lower()}_generated.mid",
                mime="audio/midi"
            )
        
        # Visualize note distribution
        note_counts = pd.DataFrame({
            'Note': list(set(generated_notes)),
            'Count': [generated_notes.count(note) for note in set(generated_notes)]
        })
        
        fig = px.bar(note_counts, x='Note', y='Count', 
                    title=f"Note Distribution - {genre} Style",
                    color='Count', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

def training_mode():
    """Custom model training interface"""
    st.markdown('<div class="section-header">🔧 Train Custom Model</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Data Upload", "🏋️ Training", "🎵 Generate"])
    
    with tab1:
        st.markdown("### Upload Your MIDI Files")
        uploaded_files = st.file_uploader(
            "Choose MIDI files", 
            type=['mid', 'midi'], 
            accept_multiple_files=True,
            help="Upload MIDI files to train on. For demo purposes, we'll use sample data."
        )
        
        use_sample_data = st.checkbox("Use sample training data", value=True)
        
        if st.button("📊 Process Data"):
            process_training_data(uploaded_files, use_sample_data)
    
    with tab2:
        if st.session_state.processor is None:
            st.info("👆 Please process data first in the 'Data Upload' tab")
        else:
            training_interface()
    
    with tab3:
        if not st.session_state.trained:
            st.info("🏋️ Please train the model first!")
        else:
            custom_generation_interface()

def process_training_data(uploaded_files, use_sample_data):
    """Process training data"""
    with st.spinner("📊 Processing training data..."):
        processor = MusicDataProcessor()
        
        if use_sample_data or not uploaded_files:
            # Use sample data
            notes = processor.create_sample_dataset()
            st.success(f"✅ Created sample dataset with {len(notes)} notes")
        else:
            # Process uploaded files
            all_notes = []
            for file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
                    tmp.write(file.read())
                    tmp.flush()
                    file_notes = processor.extract_notes_from_midi(tmp.name)
                    all_notes.extend(file_notes)
                    os.unlink(tmp.name)
            
            notes = all_notes
            st.success(f"✅ Processed {len(uploaded_files)} files with {len(notes)} total notes")
        
        # Prepare sequences
        X, y = processor.prepare_sequences(notes)
        
        # Store in session state
        st.session_state.processor = processor
        st.session_state.X = X
        st.session_state.y = y
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Notes", len(notes))
        with col2:
            st.metric("Unique Notes", len(processor.note_to_int))
        with col3:
            st.metric("Training Sequences", len(X))

def training_interface():
    """Model training interface"""
    st.markdown("### 🏋️ Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        epochs = st.slider("🔄 Training Epochs", 5, 100, 20)
        batch_size = st.selectbox("📦 Batch Size", [32, 64, 128], index=1)
        
    with col2:
        st.markdown("### ⚙️ Model Info")
        st.write(f"Vocabulary Size: {len(st.session_state.processor.note_to_int)}")
        st.write(f"Sequence Length: {st.session_state.processor.sequence_length}")
    
    if st.button("🚀 Start Training"):
        train_model(epochs, batch_size)

def train_model(epochs: int, batch_size: int):
    """Train the music generation model"""
    with st.spinner("🏋️ Training model... This may take a while!"):
        # Create model
        n_vocab = len(st.session_state.processor.note_to_int)
        generator = MusicGenerator(n_vocab)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a custom callback to update progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
        
        # Train model
        history = generator.train(
            st.session_state.X, 
            st.session_state.y, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Store in session state
        st.session_state.model = generator
        st.session_state.trained = True
        st.session_state.training_history = history
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed!")
        st.success("🎉 Model trained successfully!")
        
        # Show training metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Loss", f"{history['loss'][-1]:.4f}")
        with col2:
            st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")

def custom_generation_interface():
    """Interface for generating music with trained model"""
    st.markdown("### 🎵 Generate Custom Music")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_notes = st.slider("📝 Number of Notes to Generate", 50, 500, 200)
        temperature = st.slider("🌡️ Creativity Temperature", 0.2, 2.0, 1.0, 0.1)
        
    with col2:
        st.markdown("### 🎼 Model Stats")
        st.write(f"Vocabulary: {st.session_state.model.n_vocab} notes")
        st.write(f"Model: LSTM + Dense layers")
    
    if st.button("🎲 Generate Custom Music"):
        generate_custom_music(num_notes, temperature)

def generate_custom_music(num_notes: int, temperature: float):
    """Generate music using trained model"""
    with st.spinner("🎵 Generating your custom composition..."):
        # Create seed sequence
        seed_sequence = list(range(100))  # Use first 100 integers as seed
        
        # Generate music
        generated_indices = st.session_state.model.generate_music(
            seed_sequence, num_notes, temperature
        )
        
        # Convert to notes
        processor = st.session_state.processor
        generated_notes = [processor.int_to_note[idx % len(processor.int_to_note)] 
                          for idx in generated_indices]
        
        # Create MIDI
        midi_file = MIDIConverter.create_midi_from_notes(generated_notes, "custom_generated.mid")
        
        st.success("🎉 Custom music generated!")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Notes Generated", len(generated_notes))
        with col2:
            st.metric("Unique Notes Used", len(set(generated_notes)))
        with col3:
            st.metric("Temperature Used", temperature)
        
        # Download
        with open(midi_file, "rb") as f:
            st.download_button(
                label="📥 Download Custom MIDI",
                data=f.read(),
                file_name="custom_ai_composition.mid",
                mime="audio/midi"
            )
        
        # Visualization
        visualize_generated_music(generated_notes)

def visualize_generated_music(notes: List[str]):
    """Visualize generated music"""
    # Note frequency
    note_counts = {}
    for note in notes:
        note_counts[note] = note_counts.get(note, 0) + 1
    
    # Create DataFrame
    df = pd.DataFrame(list(note_counts.items()), columns=['Note', 'Frequency'])
    df = df.sort_values('Frequency', ascending=True)
    
    # Plot
    fig = px.bar(df.tail(10), x='Frequency', y='Note', orientation='h',
                title="Top 10 Most Used Notes", 
                color='Frequency', color_continuous_scale='plasma')
    st.plotly_chart(fig, use_container_width=True)
    
    # Note sequence visualization (first 50 notes)
    sequence_df = pd.DataFrame({
        'Position': range(min(50, len(notes))),
        'Note': notes[:50]
    })
    
    fig2 = px.line(sequence_df, x='Position', y='Note', 
                  title="Note Sequence (First 50 Notes)",
                  markers=True)
    st.plotly_chart(fig2, use_container_width=True)

def analytics_mode():
    """Model analytics and insights"""
    st.markdown('<div class="section-header">📊 Model Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.info("🏋️ Train a model first to see analytics!")
        return
    
    # Training history
    if st.session_state.training_history:
        history = st.session_state.training_history
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss plot
            epochs = range(1, len(history['loss']) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(epochs), y=history['loss'], 
                                   mode='lines', name='Training Loss'))
            fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], 
                                   mode='lines', name='Validation Loss'))
            fig.update_layout(title='Training Loss Over Time', 
                            xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(epochs), y=history['accuracy'], 
                                    mode='lines', name='Training Accuracy'))
            fig2.add_trace(go.Scatter(x=list(epochs), y=history['val_accuracy'], 
                                    mode='lines', name='Validation Accuracy'))
            fig2.update_layout(title='Model Accuracy Over Time', 
                             xaxis_title='Epoch', yaxis_title='Accuracy')
            st.plotly_chart(fig2, use_container_width=True)
    
    # Model architecture
    st.markdown("### 🏗️ Model Architecture")
    if st.session_state.model and st.session_state.model.model:
        # Display model summary in a more readable format
        model_info = {
            "Layer": [],
            "Type": [],
            "Output Shape": [],
            "Parameters": []
        }
        
        for layer in st.session_state.model.model.layers:
            model_info["Layer"].append(layer.name)
            model_info["Type"].append(type(layer).__name__)
            model_info["Output Shape"].append(str(layer.output_shape))
            model_info["Parameters"].append(layer.count_params())
        
        df = pd.DataFrame(model_info)
        st.dataframe(df, use_container_width=True)
        
        total_params = sum(model_info["Parameters"])
        st.metric("Total Parameters", f"{total_params:,}")

if __name__ == "__main__":
    main()
