import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# קריאת ה-API Key מ-Environment Variable
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/ivrit-ai/whisper-v2-d3-e3"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def transcribe_with_hf(file_path):
    """
    שליחת קובץ אודיו ל-Hugging Face Inference API עם פרמטר timestamps.
    """
    with open(file_path, "rb") as f:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=f,
            params={"return_timestamps": "true"}  # פרמטר לתמיכה בקבצים גדולים
        )
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()}

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    נקודת קצה API להעלאת קובץ אודיו וקבלת תמלול.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['file']
    file_path = os.path.join("uploads", audio_file.filename)
    os.makedirs("uploads", exist_ok=True)  # יצירת תיקיית העלאות אם אינה קיימת
    audio_file.save(file_path)

    # שליחת הקובץ ל-Hugging Face API
    result = transcribe_with_hf(file_path)

    # מחיקת קובץ זמני
    os.remove(file_path)

    return jsonify(result)

if __name__ == "__main__":
    # הפעלת היישום
    app.run(host="0.0.0.0", port=5000)
