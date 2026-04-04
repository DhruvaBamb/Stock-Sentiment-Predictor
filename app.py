import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model config
MODEL_ID = os.getenv("MODEL_ID", "yiyanghkust/finbert-tone")

# Global pipeline object
sentiment_pipeline = None

def load_model():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            print("Loading model...")
            model = ORTModelForSequenceClassification.from_pretrained(
                "philschmid/finbert-tone-onnx"  
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "yiyanghkust/finbert-tone"
            )
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer
            )
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
        # yiyanghkust/finbert-tone returns 'Positive', 'Negative', 'Neutral'
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
