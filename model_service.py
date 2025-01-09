from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelService:
    def __init__(self):
        # Use environment variable or default to the same directory as this script
        model_path = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'model/graphology_model.h5'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print(f"Loading model from: {model_path}")
        try:
            self.model = load_model(model_path)
            self.input_shape = self.model.input_shape  # Capture the expected input shape
            print(f"Model loaded successfully with input shape: {self.input_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {str(e)}")
        
    def preprocess_image(self, image_data):
        """
        Preprocess the input image to match the model's expected input shape.
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            
            # Resize image to match the model's input dimensions
            target_size = (128, 128)  # Fixed size based on model input shape
            image = image.resize(target_size)
            
            # Convert image to grayscale (1 channel)
            image = image.convert("L")
            
            # Convert image to numpy array and normalize pixel values
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch and channel dimensions to match (1, 128, 128, 1)
            img_array = np.expand_dims(img_array, axis=(0, -1))
            return img_array
        except Exception as e:
            raise ValueError(f"Error in image preprocessing: {str(e)}")
    
    def predict(self, image_data):
        """
        Perform prediction using the model on the provided image data.
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)
            
            # Make a prediction
            prediction = self.model.predict(processed_image)
            return prediction.tolist()
        except ValueError as e:
            # Handle preprocessing-related errors
            raise ValueError(f"Preprocessing error: {str(e)}")
        except Exception as e:
            # Handle prediction-related errors
            raise RuntimeError(f"Prediction error: {str(e)}")
