from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import cv2
from combined_images import main `

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/combine-images', methods=['POST'])
def combine_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please provide two images."}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    # Save the uploaded files
    image1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
    image2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
    image1.save(image1_path)
    image2.save(image2_path)

    # Define output path
    output_path = os.path.join(OUTPUT_FOLDER, 'combined_image.png')

    # Call the GAN processing function
    success, message = main(image1_path, image2_path, output_path)

    if success:
        return send_file(output_path, mimetype='image/png')
    else:
        return jsonify({"error": message}), 500

@app.route('/')
def index():
    return

if __name__ == '__main__':
    app.run(debug=True)
