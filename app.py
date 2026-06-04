# app.py
import io
import numpy as np
import torch
import librosa
from pydub import AudioSegment
from flask import Flask, request, jsonify, send_from_directory
from transformer import EmotionRecognitionTransformer
import os
if os.name == "nt": os.environ["PATH"] += os.pathsep + os.path.dirname(__file__)

app = Flask(__name__, static_folder='.')

device = torch.device("cpu")
model = EmotionRecognitionTransformer(input_dim=117).to(device)
model.load_state_dict(torch.load("model-demo.pth", map_location="cpu"))
model.eval()

n_mels = 117
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def extract_mfcc(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    y, sr = librosa.load(wav_io, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39, n_fft=400, hop_length=160) 
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.concatenate((mfcc, delta, delta2), axis=0)  # shape: (117, T)
    features = features.T  # → (T, 117)

    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)

    # Padding to fixed length
    padded = np.zeros((1000, 117), dtype=np.float32)
    length = min(1000, len(features))
    padded[:length] = features[:length]

    tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
    mask = torch.ones(1, 1000, dtype=torch.bool).to(device)
    mask[0, length:] = True   
    mask[0, :length] = False   

    return tensor, mask

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    audio_bytes = request.files["file"].read()
    try:
        mfcc, mask = extract_mfcc(audio_bytes)
        with torch.no_grad():
            logits = model(mfcc, mask)
            pred_idx = logits.argmax(1).item()
        return jsonify({"emotion": emotions[pred_idx]})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "處理失敗"}), 500

if __name__ == "__main__":
    print("語音情感分析器已啟動！")
    print("請打開： http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)