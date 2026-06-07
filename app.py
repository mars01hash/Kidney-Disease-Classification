import os
import sys
import subprocess
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    try:
        enable_training = os.environ.get("ENABLE_TRAINING_ROUTE", "false").lower() == "true"
        if not enable_training:
            return jsonify({
                "status": "error",
                "message": "Training route is disabled. Set ENABLE_TRAINING_ROUTE=true to enable it."
            }), 403

        # Run pipeline
        result = subprocess.run(["dvc", "repro"], capture_output=True, text=True, check=True)
        return jsonify({
            "status": "success",
            "message": "Training completed successfully!",
            "output": result.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": f"DVC reproduction failed: {e.stderr or e.stdout}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        if request.is_json:
            image_data = request.json.get("image")
            if not image_data:
                return jsonify({"status": "error", "message": "No image data found in request"}), 400
            decodeImage(image_data, "inputImage.jpg")
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400
            file.save("inputImage.jpg")
        else:
            return jsonify({"status": "error", "message": "Unsupported request format. Use JSON or multipart/form-data file upload."}), 400

        pipeline = PredictionPipeline("inputImage.jpg")
        prediction = pipeline.predict()
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

