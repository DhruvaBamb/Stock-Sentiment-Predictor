import os
import numpy as np
import requests
from transformers import AutoTokenizer
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_DIR   = "/tmp/finbert"
ONNX_PATH   = os.path.join(MODEL_DIR, "model.onnx")

# HuggingFace raw URLs for ProsusAI/finbert
# model.onnx is inside the onnx/ subfolder
ONNX_URL  = "https://huggingface.co/ProsusAI/finbert/resolve/main/onnx/model.onnx"

LABELS = ["positive", "negative", "neutral"]   # finbert label order

# ── Globals ────────────────────────────────────────────────────────────────────
tokenizer   = None
ort_session = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def _download(url, dest):
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} → {dest}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    print(f"Saved {dest} ({len(r.content)//1024} KB)")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ── Load model on startup ──────────────────────────────────────────────────────
def load_model():
    global tokenizer, ort_session
    try:
        print("Initializing tokenizer and downloading ONNX model …")
        
        # Use AutoTokenizer to handle config/vocab automatically
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        # Download ONNX model for inference
        _download(ONNX_URL, ONNX_PATH)

        ort_session = ort.InferenceSession(
            ONNX_PATH,
            providers=["CPUExecutionProvider"]
        )
        print("Model loaded successfully ✓")
    except Exception as e:
        print(f"Error loading model: {e}")
        tokenizer   = None
        ort_session = None

load_model()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": ort_session is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if ort_session is None or tokenizer is None:
        return jsonify({"error": "Model failed to load. Check logs."}), 500

    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Tokenize using transformers AutoTokenizer
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        
        # Prepare inputs for ONNX (convert to int64)
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

        outputs = ort_session.run(None, ort_inputs)
        
        logits = outputs[0][0]
        probs  = softmax(logits)
        idx    = int(np.argmax(probs))

        return jsonify({
            "text":       text,
            "prediction": LABELS[idx].capitalize(),
            "confidence": round(float(probs[idx]) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
