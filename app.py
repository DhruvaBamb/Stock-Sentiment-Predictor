import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

HF_TOKEN   = os.getenv("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert/pipeline/text-classification"
HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}"}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "api_configured": HF_TOKEN is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        if not HF_TOKEN:
            return jsonify({"error": "HF_TOKEN not set in environment."}), 500
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text}, timeout=30)
        if response.status_code == 503:
            return jsonify({"error": "Model is warming up on HuggingFace, retry in 20 seconds."}), 503
        response.raise_for_status()
        results = response.json()
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]
        best = max(results, key=lambda x: x["score"])
        return jsonify({"text": text, "prediction": best["label"].capitalize(), "confidence": round(best["score"] * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
