import os
import base64
import tempfile
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)

# Connect to your Hugging Face Space
client = Client("chemoiko/banana-resnet-5000")
API_NAME = "/predict"

@app.route('/')
def hello_world():
    return '🍌 Banana disease API is running!'

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    tmp_path = None
    try:
        # Get JSON from Flutter
        data = request.get_json()
        if "image_base64" not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode the base64 image into bytes and save temporarily
        image_bytes = base64.b64decode(data["image_base64"])
        
        # Use tempfile for better handling on Render
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        # Send image to Hugging Face Space
        result = client.predict(
            img=handle_file(tmp_path),
            api_name=API_NAME
        )
        
        # Return prediction to Flutter
        return jsonify({"prediction": result[0], "details": result[1]})
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error for debugging
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Delete temp file (even if error occurs)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
