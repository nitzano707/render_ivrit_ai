import os
import requests
from flask import Flask, request, jsonify
from pydub import AudioSegment

app = Flask(__name__)

# קריאת ה-API Key מ-Environment Variable
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/ivrit-ai/whisper-v2-d3-e3"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def transcribe_with_hf(file_path):
    """
    שליחת קובץ אודיו ל-Hugging Face Inference API עם חותמות זמן.
    """
    with open(file_path, "rb") as f:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=f,
            params={"return_timestamps": "true"}  # בקשה להחזיר חותמות זמן
        )
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()}

def split_audio(file_path, segment_length=30000):
    """
    חיתוך קובץ אודיו למקטעים של עד 30 שניות (30000ms).
    """
    audio = AudioSegment.from_file(file_path)
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        segment_path = f"{file_path}_part{i // segment_length}.wav"
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
    return segments

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    נקודת קצה API להעלאת קובץ אודיו וקבלת תמלול עם חותמות זמן.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['file']
    file_path = os.path.join("uploads", audio_file.filename)
    os.makedirs("uploads", exist_ok=True)
    audio_file.save(file_path)

    # חיתוך קובץ ארוך
    segments = split_audio(file_path)

    # תמלול כל המקטעים
    full_transcription = []
    timestamps = []
    for segment_path in segments:
        result = transcribe_with_hf(segment_path)
        if "segments" in result:
            for segment in result["segments"]:
                full_transcription.append(segment["text"])
                timestamps.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                })
        os.remove(segment_path)  # מחיקת המקטע

    # מחיקת הקובץ המקורי
    os.remove(file_path)

    return jsonify({"transcription": " ".join(full_transcription), "timestamps": timestamps}, ensure_ascii=False)

if __name__ == "__main__":
    # הפעלת היישום
    app.run(host="0.0.0.0", port=5000)
