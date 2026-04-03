import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model config
MODEL_ID = os.getenv("MODEL_ID", "ProsusAI/finbert")

# Global pipeline object
sentiment_pipeline = None

def load_model():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            print(f"Loading model {MODEL_ID}...")
            # We use text-classification/sentiment-analysis pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_ID)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

@app.before_request
def before_first_request_func():
    # Load model on first request to save start time (alternatively can load globally outside)
    load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if not sentiment_pipeline:
        load_model()
        
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in JSON payload"}), 400
        
    text = data["text"].strip()
    if not text:
         return jsonify({"error": "Text cannot be empty"}), 400
         
    try:
        # Run inference
        results = sentiment_pipeline(text)
        
        # results format: [{'label': 'positive', 'score': 0.94...}]
        # ProsusAI/finbert returns 'positive', 'negative', 'neutral'
        result = results[0]
        
        return jsonify({
            "text": text,
            "prediction": result['label'],
            "confidence": result['score']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
