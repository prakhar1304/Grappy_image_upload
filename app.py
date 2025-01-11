from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime
from werkzeug.utils import secure_filename
from model_service import ModelService

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Initialize model service
model_service = ModelService()

# AWS S3 configuration
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# MongoDB configuration
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client['image_db']
images_collection = db['images']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting personality traits."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "File is empty"}), 400

    try:
        image_data = file.read()
        prediction = model_service.predict(image_data)
        return jsonify({"success": True, "prediction": prediction}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        return jsonify({"error": "Unexpected error occurred"}), 500

@app.route('/images', methods=['GET'])
def get_images():
    """Fetch all uploaded images metadata from MongoDB."""
    try:
        images = list(images_collection.find())
        for image in images:
            image['_id'] = str(image['_id'])  # Convert ObjectId to string
        return jsonify(images)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image to S3 and save metadata in MongoDB."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file format'}), 400

    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"

        # Upload to S3
        s3.upload_fileobj(
            file,
            os.getenv('S3_BUCKET_NAME'),
            unique_filename
        )

        # Generate S3 URL
        file_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{unique_filename}"

        # Save metadata in MongoDB
        images_collection.insert_one({
            'filename': unique_filename,
            'original_filename': filename,
            'url': file_url,
            'uploaded_at': datetime.now()
        })

        return jsonify({'success': True, 'message': 'File uploaded successfully', 'url': file_url})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
