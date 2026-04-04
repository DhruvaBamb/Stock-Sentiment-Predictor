import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# At top of file - load model on startup
sentiment_pipeline = None

def load_model():
    global sentiment_pipeline
    try:
        print("Loading model...")
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
        
        model = ORTModelForSequenceClassification.from_pretrained(
            "optimum/finbert-sentiment"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert"
        )
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sentiment_pipeline = None

# Load on startup
load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": sentiment_pipeline is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    # Check if model loaded successfully
    if sentiment_pipeline is None:
        return jsonify({
            "error": "Model failed to load. Check logs."
        }), 500
        
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
         
    try:
        results = sentiment_pipeline(text)
        result = results[0]
        return jsonify({
            "text": text,
            "prediction": result['label'],
            "confidence": round(result['score'] * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
