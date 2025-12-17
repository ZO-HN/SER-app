# speech-emotion-recognition-app

# CNN-LSTM Speech Emotion Recognition System

## 🎯 Overview

This updated system uses a **CNN-LSTM hybrid model** with a comprehensive multi-modal feature extraction pipeline. The model analyzes speech through 262 distinct features across four domains: acoustic (MFCCs), spectral (Mel-Spectrograms), spectral characteristics, and prosodic patterns.

---

## 📋 What Changed

### 1. **Model Architecture**
- **Old:** Simple CNN with Mel-Spectrograms only (128 features)
- **New:** CNN-LSTM with multi-modal features (262 features)
  - MFCCs + Deltas: 120 features
  - Mel-Spectrograms: 128 features
  - Spectral Features: 12 features
  - Prosodic Features: 2 features

### 2. **Input Shape**
- **Old:** `(1, 120, 150, 1)` - Single feature type
- **New:** `(1, 262, 150, 1)` - Stacked multi-modal features

### 3. **Prediction Accuracy**
- **Improvement:** The new model should provide more robust predictions by analyzing multiple aspects of speech simultaneously
- **Better at:** Detecting subtle emotional cues through prosodic analysis
- **Benefit:** Reduced "hallucination" on complex audio samples

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy librosa pandas streamlit soundfile matplotlib scipy
```

### File Structure

```
project/
├── models/
│   └── cnn-lstm/
│       └── best-cnn-lstm-ser-model.keras  # Your trained model
├── validation_samples/                     # Optional test audio files
├── inference.py                            # Updated feature extraction
├── web_app_cnn_lstm.py                     # Updated Streamlit app
└── README.md                               # This file
```

### Running the Application

```bash
streamlit run web_app_cnn_lstm.py
```

---

## 🔧 Technical Details

### Feature Extraction Pipeline

The system extracts features in this exact order to match training:

#### 1. **MFCCs with Deltas** (120 features)
```python
- Base MFCCs: 40 coefficients
- Delta (1st derivative): 40 coefficients  
- Delta-Delta (2nd derivative): 40 coefficients
Total: 120 features
```

#### 2. **Mel-Spectrogram** (128 features)
```python
- 128 mel-frequency bins
- Power-to-dB conversion
- Captures frequency distribution
```

#### 3. **Spectral Features** (12 features)
```python
- Spectral Centroid: 1
- Spectral Bandwidth: 1
- Spectral Contrast: 7
- Spectral Flatness: 1
- Spectral Rolloff: 1
- Zero Crossing Rate: 1
Total: 12 features
```

#### 4. **Prosodic Features** (2 features)
```python
- F0 (Fundamental Frequency/Pitch): 1
- RMS (Root Mean Square Energy): 1
Total: 2 features
```

### Processing Steps

1. **Load Audio** → 16kHz sample rate
2. **Extract All Features** → 262 feature vectors
3. **Time Alignment** → Trim all to minimum time dimension
4. **Stack Vertically** → Concatenate all feature types
5. **Pad/Trim** → Fixed length of 150 time frames
6. **Normalize** → Z-score normalization
7. **Add Dimensions** → Shape becomes (1, 262, 150, 1)
8. **Predict** → Pass to CNN-LSTM model

---

## 🎯 Key Improvements in Code

### 1. **Robust Session State Management**
```python
# All variables initialized at startup
initialize_session_state()

# Safe access - no more KeyError
emotion = st.session_state.predicted_emotion  # Always exists
```

### 2. **Enhanced Silence Detection**
```python
def is_audio_silent(audio, sr=16000, silence_threshold=0.01):
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    is_silent = rms < silence_threshold or zcr < 0.01
    return is_silent, rms
```

### 3. **Complete Feature Pipeline**
```python
# Exact match to training pipeline
features = np.vstack([
    mfccs,           # (120, T)
    mel_spectrogram, # (128, T)
    spectral,        # (12, T)
    prosodic         # (2, T)
])
# Result: (262, T)
```

### 4. **Error Handling**
```python
try:
    result = predictor.predict_with_details(audio_path)
    # Success path
except ValueError as ve:
    # Validation errors (duration, etc.)
except Exception as e:
    # General errors
```

---

## 📊 Model Performance

Based on EmoDB dataset validation:

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 52.86% |
| **Precision** | 0.63 |
| **Recall** | 0.53 |
| **F1-Score** | 0.52 |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.80 | 0.68 | 0.74 |
| Disgust | 0.50 | 0.22 | 0.30 |
| Fear | 0.34 | **0.93** | 0.50 |
| Happy | 0.51 | 0.28 | 0.36 |
| Neutral | **0.89** | 0.30 | 0.45 |
| Sad | 0.50 | 0.58 | 0.54 |

**Strengths:**
- ✅ Excellent at detecting Fear (93% recall)
- ✅ High precision for Neutral (89%)
- ✅ Strong Angry detection (80% precision)

**Weaknesses:**
- ⚠️ Struggles with Disgust (22% recall)
- ⚠️ Happy emotion is challenging (28% recall)

---

## 🐛 Common Issues & Solutions

### Issue 1: Model File Not Found
```
Error: No file or directory: 'models/cnn-lstm/best-cnn-lstm-ser-model.keras'
```

**Solution:**
```bash
# Ensure correct path
mkdir -p models/cnn-lstm
# Copy your model file
cp /path/to/best-cnn-lstm-ser-model.keras models/cnn-lstm/
```

### Issue 2: Feature Shape Mismatch
```
Error: Input shape mismatch. Expected (262, 150, 1), got (261, 150, 1)
```

**Solution:**
The training code shows 262 features. If you get 261, one feature extraction might be configured differently. Check:
- Spectral contrast bands (should be 7)
- All feature types are included

### Issue 3: Silence Still Causing Issues
```
Warning: Model predicts with high confidence on silent audio
```

**Solution:**
Adjust threshold in sidebar settings or code:
```python
silence_threshold = 0.02  # Increase for stricter filtering
```

### Issue 4: Slow Predictions
**Cause:** Feature extraction is computationally intensive

**Solution:**
- Ensure audio is < 10 seconds
- Use GPU-enabled TensorFlow if available
- Consider batch processing for multiple files

---

## 🔍 Debugging Tips

### 1. Check Feature Shapes
```python
result = predictor.predict_with_details(audio_path)
print(f"Feature shape: {result['feature_shape']}")
print(f"Audio duration: {result['audio_duration']}")
```

### 2. Inspect Extracted Features
```python
from inference import InferenceAudioGenerator

gen = InferenceAudioGenerator()
features, audio = gen.generate("test.wav")
print(f"Features: {features.shape}")  # Should be (1, 262, 150, 1)
```

### 3. Validate Model Input
```python
print(f"Model input shape: {predictor.model.input_shape}")
print(f"Model output shape: {predictor.model.output_shape}")
```

---

## 📈 Expected Behavior

### Good Predictions
- **Clear speech:** 60-90% confidence
- **Strong emotions:** Higher confidence (Angry, Fear)
- **Neutral speech:** 50-70% confidence

### Expected Low Confidence
- **Mixed emotions:** 40-60%
- **Subtle expressions:** 40-55%
- **Background noise:** May reduce confidence

### Should Reject (Silence Detection)
- **Pure silence:** RMS < 0.01
- **Very quiet audio:** RMS < 0.005
- **No voice activity:** ZCR < 0.01

---

## 🎓 Understanding the Architecture

### CNN Component
```
Input (262, 150, 1)
    ↓
Conv2D Blocks × 3
    ↓
Spatial Features Extracted
    ↓
Reshape for LSTM
```

### LSTM Component
```
Reshaped Features
    ↓
LSTM Layer 1 (256 units) - Temporal patterns
    ↓
LSTM Layer 2 (128 units) - Refined patterns
    ↓
Dense Layers - Classification
    ↓
Softmax Output (6 emotions)
```

### Why This Works
1. **CNN extracts spatial patterns** from stacked features
2. **LSTM captures temporal dynamics** of emotional expression
3. **Multi-modal features** provide redundancy and robustness
4. **Prosodic analysis** adds emotional tone information

---

## 🔄 Migration from Old System

### Code Changes Required

**Old Code:**
```python
from inference import CNNAudioPredictor

predictor = CNNAudioPredictor(
    model_path="models/cnn/best_cnn-ser-model.keras",
    class_names=EMOTIONS
)
```

**New Code:**
```python
from inference import CNNLSTMAudioPredictor

predictor = CNNLSTMAudioPredictor(
    model_path="models/cnn-lstm/best-cnn-lstm-ser-model.keras",
    class_names=EMOTIONS
)
```

### Feature Extraction Changes

**Old:** Simple mel-spectrogram
**New:** Complete pipeline (automatic in `CNNLSTMAudioPredictor`)

No changes needed in application code - the predictor handles everything!

---

## 📝 API Reference

### CNNLSTMAudioPredictor

#### `__init__(model_path, class_names)`
Initialize predictor with model and emotion labels.

#### `predict(file_path)`
Predict emotion from audio file.

**Returns:**
```python
{
    "emotion": str,              # Predicted emotion
    "confidence": float,         # 0-1 confidence score
    "all_probabilities": dict,   # All class probabilities
    "audio": np.array           # Raw audio for visualization
}
```

#### `predict_with_details(file_path)`
Predict with additional diagnostic information.

**Returns:** Same as `predict()` plus:
```python
{
    "feature_shape": tuple,      # Shape of extracted features
    "audio_duration": float      # Duration in seconds
}
```

---

## 🤝 Support

If you encounter issues:

1. **Check model file exists and path is correct**
2. **Verify all dependencies are installed**
3. **Test with a known good audio file**
4. **Check feature extraction shapes match training**
5. **Review error messages in terminal**

---

## 📄 License

This code follows the same license as your original SER project.

---

## ✨ Next Steps

1. **Test the system** with various audio samples
2. **Monitor prediction quality** - confidence scores should be more consistent
3. **Collect problematic samples** for future model improvement
4. **Consider fine-tuning** on domain-specific data
5. **Implement audio augmentation** if needed for better generalization

---

**Version:** 2.0 (CNN-LSTM)  
**Last Updated:** December 2024  
**Compatibility:** TensorFlow 2.x, Python 3.8+

