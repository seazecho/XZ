from flask import Flask, request, jsonify
from transformers import pipeline
import librosa
import numpy as np
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

app = Flask(__name__)

# # Load Text Emotion Model (RoBERTa)
# text_model = pipeline('text-classification', model="SamLowe/roberta-base-go_emotions", top_k=None)

# Load Audio Emotion Model (SVM trained on MFCC features)
audio_model = joblib.load("ser_model.pkl")

# Emotion labels for audio model
EMOTION_LABELS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# def classify_text_emotion(text):
#     """Get emotion from text using RoBERTa"""
#     result = text_model(text)[0]
#     return max(result, key=lambda x: x['score'])['label']


# Load emotion classification model
logging.info("Loading emotion classification model...")
emotion_classifier = pipeline('text-classification', model="SamLowe/roberta-base-go_emotions", top_k=None)

def classify_emotion_X(text):
    """Classify emotion from the given text."""
    try:
        results = emotion_classifier(text)[0]
        max_score = max(results, key=lambda x: x['score'])
        return max_score['label'] if max_score['score'] >= 0.5 else 'neutral'
    except Exception as e:
        logging.error(f"Error classifying emotion: {e}")
        return 'unknown'

@app.route('/classify_text', methods=['POST'])
def classify_emotion_endpoint():
    """API endpoint to classify emotion from transcribed text."""
    try:
        # Extract text from the JSON request body
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Classify emotion
        emotion = classify_emotion_X(text)
        return jsonify({'text': text, 'emotion': emotion})
    except Exception as e:
        logging.error(f"Error in API: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def extract_audio_features(file_path):
    """Extract MFCC, pitch, energy, ZCR from audio file"""
    signal, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    pitch = np.max(librosa.piptrack(y=signal, sr=sr), axis=0)
    energy = librosa.feature.rms(y=signal)
    zcr = librosa.feature.zero_crossing_rate(y=signal)

    # Convert features into a format suitable for SVM model
    features = np.hstack([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(pitch), np.std(pitch),
        np.mean(energy), np.std(energy),
        np.mean(zcr), np.std(zcr)
    ])
    return features

def classify_audio_emotion(file_path):
    """Predict emotion from audio using the trained SVM model"""
    features = extract_audio_features(file_path)
    emotion_code = audio_model.predict([features])[0]
    return EMOTION_LABELS[emotion_code]

@app.route('/classify_audio', methods=['POST'])
def classify_emotion_Z():
    """API endpoint to classify emotions from text and/or audio"""
    # text = request.form.get("text", "")
    file = request.files.get("audio")

    # text_emotion = classify_text_emotion(text) if text else "unknown"
    audio_emotion = "unknown"

    if file:
        file_path = "temp_audio.wav"
        file.save(file_path)
        audio_emotion = classify_audio_emotion(file_path)
        os.remove(file_path)  # Clean up

    return jsonify({"audio_emotion": audio_emotion})

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from Railway
    app.run(debug=True, host="0.0.0.0", port=port)
