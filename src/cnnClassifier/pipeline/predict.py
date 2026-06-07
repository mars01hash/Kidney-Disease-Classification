import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model_path = Path("artifacts/training/model.h5")
        model = tf.keras.models.load_model(model_path)

        # Preprocess image
        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Rescale the image matching the training preprocessing
        test_image = test_image / 255.0

        # Run inference
        preds = model.predict(test_image)
        result = np.argmax(preds, axis=1)[0]

        # Load class indices
        class_indices_path = Path("artifacts/training/class_indices.json")
        if class_indices_path.exists():
            with open(class_indices_path, "r") as f:
                class_indices = json.load(f)
            # Invert mapping: index -> class name
            inv_map = {v: k for k, v in class_indices.items()}
            prediction = inv_map.get(result, "Normal")
        else:
            # Fallback mapping
            prediction = "Tumor" if result == 1 else "Normal"

        return [{"image": prediction}]

