from flask import Flask, request, jsonify
import os
from pydub import AudioSegment
import json

app = Flask(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/ivrit-ai/whisper-v2-d3-e3"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def transcribe_with_hf(file_path):
    with open(file_path, "rb") as f:
        response = requests.post(API_URL, headers=HEADERS, data=f)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()}

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['file']

    # בדיקת סוג הקובץ
    if not audio_file.filename.endswith(('.wav', '.mp3', '.mp4')):
        return jsonify({"error": "Unsupported file format. Please upload a .wav, .mp3, or .mp4 file."}), 400

    # שמירת קובץ
    file_path = os.path.join("uploads", audio_file.filename)
    os.makedirs("uploads", exist_ok=True)
    audio_file.save(file_path)

    # המרת MP4 ל-WAV (אם צריך)
    if file_path.endswith(".mp4"):
        audio = AudioSegment.from_file(file_path, format="mp4")
        audio.export(file_path, format="wav")

    # תמלול
    result = transcribe_with_hf(file_path)

    # מחיקת קובץ זמני
    os.remove(file_path)

    # הכנת תגובה
    response = {"transcription": result.get("text", ""), "timestamps": result.get("segments", [])}
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False),
        status=200,
        mimetype="application/json"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
