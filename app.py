import os
import base64
import tempfile
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

# -------------------------
# CONFIG
# -------------------------
# Hugging Face Space
HF_SPACE = "chemoiko/banana-disease-api"
API_NAME = "/predict"
HF_TOKEN = os.environ.get("HF_TOKEN")  # if your space is private

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)
client = Client(HF_SPACE, hf_token=HF_TOKEN)

@app.route('/')
def hello_world():
    return '🍌 Banana Disease API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 and save temp file
        image_bytes = base64.b64decode(data["image_base64"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        # Send image to Hugging Face Space
        result = client.predict(
            img=handle_file(tmp_path),
            api_name=API_NAME
        )

        # Cleanup temp file
        os.remove(tmp_path)

        # Return prediction
        return jsonify({"prediction": result[0], "details": result[1]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
