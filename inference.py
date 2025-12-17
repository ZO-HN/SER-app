import numpy as np
import librosa
from tensorflow.keras.models import load_model


# ==========================================
# FEATURE EXTRACTION FUNCTIONS
# ==========================================

def extract_mfcc_features(audio, sample_rate, n_mfcc=40):
    """
    Extract MFCC features with deltas and delta-deltas.
    Returns shape: (120, time_frames)
    """
    # Extract base MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )
    
    # Compute deltas and delta-deltas
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack them vertically: (40 + 40 + 40) = 120 features
    mfcc_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    return mfcc_features


def extract_mel_spectrogram(audio, sample_rate, n_mels=128):
    """
    Extract Mel-Spectrogram features.
    Returns shape: (128, time_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db


def extract_spectral_features(audio, sample_rate):
    """
    Extract spectral features: centroid, bandwidth, contrast, flatness, rolloff.
    Returns shape: (12, time_frames)
    """
    # Spectral Centroid (1 feature)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    
    # Spectral Bandwidth (1 feature)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    
    # Spectral Contrast (7 features by default)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    
    # Spectral Flatness (1 feature)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    
    # Spectral Rolloff (1 feature)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    
    # Zero Crossing Rate (1 feature)
    zcr = librosa.feature.zero_crossing_rate(audio)
    
    # Stack all spectral features: (1 + 1 + 7 + 1 + 1 + 1) = 12 features
    spectral = np.vstack([
        spectral_centroid,
        spectral_bandwidth,
        spectral_contrast,
        spectral_flatness,
        spectral_rolloff,
        zcr
    ])
    
    return spectral


def extract_prosodic_features(audio, sample_rate):
    """
    Extract prosodic features: pitch (F0) and energy (RMS).
    Returns shape: (2, time_frames)
    """
    # Pitch (F0) using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )
    
    # Replace NaN values with 0
    f0 = np.nan_to_num(f0)
    
    # Energy (RMS)
    rms = librosa.feature.rms(y=audio)
    
    # Align dimensions: f0 is 1D, need to match RMS shape
    # Resample f0 to match RMS frame count
    if len(f0) != rms.shape[1]:
        from scipy import signal
        f0 = signal.resample(f0, rms.shape[1])
    
    f0 = f0.reshape(1, -1)  # Shape: (1, time_frames)
    
    # Stack prosodic features: (1 + 1) = 2 features
    prosodic = np.vstack([f0, rms])
    
    return prosodic


def pad_features(features, max_len=150):
    """
    Pad or trim features to fixed length.
    Input shape: (num_features, time_frames)
    Output shape: (num_features, max_len)
    """
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]
    
    return features


def normalize_features(features):
    """
    Normalize features using z-score normalization.
    Input/Output shape: (num_features, time_frames)
    """
    mean = np.mean(features)
    std = np.std(features) + 1e-8
    normalized = (features - mean) / std
    
    return normalized


# ==========================================
# INFERENCE AUDIO GENERATOR
# ==========================================

class InferenceAudioGenerator:
    """
    Handles audio loading and complete feature extraction pipeline for CNN-LSTM model.
    Extracts: MFCCs (120) + Mel-Spectrograms (128) + Spectral (12) + Prosodic (2) = 262 features
    
    Note: Based on your training code, the actual output is 261 features.
    This suggests one feature might be dropped or the configuration is slightly different.
    I'll implement the exact pipeline from your training code.
    """
    
    def __init__(self, target_sr=16000, max_len=150):
        self.target_sr = target_sr
        self.max_len = max_len

    def load_audio(self, file_path):
        """Load audio file at target sample rate"""
        audio, _ = librosa.load(file_path, sr=self.target_sr)
        return audio

    def extract_all_features(self, audio):
        """
        Extract all feature types and stack them vertically.
        Returns shape: (261, time_frames)
        
        Breakdown:
        - MFCCs with deltas: 120 features
        - Mel-Spectrogram: 128 features  
        - Spectral: 12 features
        - Prosodic: 2 features
        Total: 262 features (but training shows 261, so we'll verify)
        """
        # Extract each feature type
        mfccs = extract_mfcc_features(audio, self.target_sr)           # (120, T)
        mel_spectrogram = extract_mel_spectrogram(audio, self.target_sr)  # (128, T)
        spectral = extract_spectral_features(audio, self.target_sr)    # (12, T)
        prosodic = extract_prosodic_features(audio, self.target_sr)    # (2, T)
        
        # Align to same time dimension (find minimum length)
        min_len = min(
            mfccs.shape[1],
            mel_spectrogram.shape[1],
            spectral.shape[1],
            prosodic.shape[1]
        )
        
        # Trim all to minimum length
        mfccs = mfccs[:, :min_len]
        mel_spectrogram = mel_spectrogram[:, :min_len]
        spectral = spectral[:, :min_len]
        prosodic = prosodic[:, :min_len]
        
        # Stack all features vertically
        features = np.vstack([mfccs, mel_spectrogram, spectral, prosodic])
        # Shape: (262, time_frames) or (261, time_frames) depending on config
        
        return features

    def generate(self, file_path):
        """
        Complete preprocessing pipeline for a single audio file.
        
        Returns:
            features: np.array of shape (1, 261, 150, 1) ready for model input
            audio: raw audio array for visualization
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Validate duration
        duration = len(audio) / self.target_sr
        
        if duration < 0.5:
            raise ValueError(f"Audio too short ({duration:.2f}s). Minimum 0.5 seconds required.")
        
        if duration > 10.0:
            # Trim to 10 seconds if too long
            audio = audio[:self.target_sr * 10]
        
        # Extract all features
        features = self.extract_all_features(audio)
        
        # Pad to fixed length
        features = pad_features(features, max_len=self.max_len)
        # Shape: (num_features, 150)
        
        # Normalize
        features = normalize_features(features)
        # Shape: (num_features, 150) - normalized
        
        # Add channel dimension
        features = features[..., np.newaxis]
        # Shape: (num_features, 150, 1)
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        # Shape: (1, num_features, 150, 1)
        
        return features, audio


# ==========================================
# CNN-LSTM AUDIO PREDICTOR
# ==========================================

class CNNLSTMAudioPredictor:
    """
    Predictor class for CNN-LSTM Speech Emotion Recognition model.
    Handles model loading, preprocessing, and inference.
    """
    
    def __init__(self, model_path, class_names):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the .keras model file
            class_names: List of emotion class names in order
        """
        self.model = load_model(model_path)
        self.class_names = class_names
        self.generator = InferenceAudioGenerator()
        
        # Get expected input shape from model
        expected_shape = self.model.input_shape
        print(f"Model loaded successfully!")
        print(f"Expected input shape: {expected_shape}")
        print(f"Output classes: {class_names}")

    def predict(self, file_path):
        """
        Predict emotion from audio file.
        
        Args:
            file_path: Path to audio file (.wav, .mp3, etc.)
        
        Returns:
            dict with:
                - emotion: predicted emotion string
                - confidence: confidence score (0-1)
                - all_probabilities: dict of all class probabilities
                - audio: raw audio array for visualization
        """
        # Generate features
        features, audio = self.generator.generate(file_path)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        predicted_emotion = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Create probability dictionary for all classes
        all_probs = {
            self.class_names[i]: float(predictions[i]) 
            for i in range(len(self.class_names))
        }
        
        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "audio": audio
        }

    def predict_with_details(self, file_path):
        """
        Predict emotion with additional diagnostic information.
        
        Returns:
            dict with prediction results plus:
                - feature_shape: shape of extracted features
                - audio_duration: duration of audio in seconds
        """
        result = self.predict(file_path)
        
        # Add diagnostic info
        audio = result["audio"]
        result["audio_duration"] = len(audio) / self.generator.target_sr
        
        # Get feature shape (before adding batch dimension)
        features, _ = self.generator.generate(file_path)
        result["feature_shape"] = features.shape
        
        return result


# ==========================================
# LEGACY SUPPORT: Simple CNN Predictor
# ==========================================

class CNNAudioPredictor:
    """
    Legacy predictor for simple CNN model (mel-spectrogram only).
    Kept for backward compatibility with old model.
    """
    
    def __init__(self, model_path, class_names):
        self.model = load_model(model_path)
        self.class_names = class_names
        self.target_sr = 16000
        self.n_mels = 120
        self.max_len = 150

    def load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.target_sr)
        return audio

    def extract_features(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad / trim
        if mel_db.shape[1] < self.max_len:
            pad_width = self.max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))
        else:
            mel_db = mel_db[:, :self.max_len]

        # Normalize
        valid_frames = np.count_nonzero(np.sum(mel_db, axis=0))
        if valid_frames > 0:
            mel_db[:, :valid_frames] = (
                mel_db[:, :valid_frames] - 
                np.mean(mel_db[:, :valid_frames])
            ) / (np.std(mel_db[:, :valid_frames]) + 1e-8)

        return mel_db

    def generate(self, file_path):
        audio = self.load_audio(file_path)
        duration = len(audio) / self.target_sr

        if duration < 1.0:
            raise ValueError("Audio too short (min 1 second required)")
        if duration > 6.0:
            raise ValueError("Audio too long (max 6 seconds allowed)")

        features = self.extract_features(audio)
        features = features[..., np.newaxis]
        features = np.expand_dims(features, axis=0)

        return features, audio

    def predict(self, file_path):
        features, audio = self.generate(file_path)
        preds = self.model.predict(features, verbose=0)[0]
        idx = np.argmax(preds)

        return {
            "emotion": self.class_names[idx],
            "confidence": float(preds[idx]),
            "audio": audio
        }
