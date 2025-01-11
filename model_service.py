import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelService:
    def __init__(self):
        # Load model
        model_path = os.getenv('MODEL_PATH', 'model/graphology_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)

        # Load label encoder
        encoder_path = os.getenv('ENCODER_PATH', 'model/label_encoder.pkl')
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")
        
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)
        print("Label encoder loaded successfully.")

    def preprocess_image(self, image_data):
        """Preprocess the input image to match the model's input shape."""
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((128, 128)).convert("L")  # Grayscale
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
            return img_array
        except Exception as e:
            raise ValueError(f"Error in image preprocessing: {str(e)}")

    def predict(self, image_data):
        """Predict the class label using the model and label encoder."""
        processed_image = self.preprocess_image(image_data)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        return self.encoder.inverse_transform([predicted_class])[0]
