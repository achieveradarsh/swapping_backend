from flask import Flask, request, jsonify, send_from_directory
import cv2
import dlib
import numpy as np
import os
from imutils import face_utils

app = Flask(__name__)

# Ensure the upload and output directories exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the pre-trained model for facial landmark detection
predictor_path = 'shape_predictor_81_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

def process_images(source_path, target_path):
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    source_rects = detector(source_gray)
    target_rects = detector(target_gray)

    if len(source_rects) == 0 or len(target_rects) == 0:
        return None

    source_landmarks = predictor(source_gray, source_rects[0])
    source_landmarks = face_utils.shape_to_np(source_landmarks)

    target_landmarks = predictor(target_gray, target_rects[0])
    target_landmarks = face_utils.shape_to_np(target_landmarks)

    def extract_face_region(image, landmarks):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(landmarks), 255)
        face = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(cv2.convexHull(landmarks))
        face_cropped = face[y:y+h, x:x+w]
        return face_cropped, (x, y, w, h), mask

    source_face, source_bbox, source_mask = extract_face_region(source_image, source_landmarks)

    x, y, w, h = cv2.boundingRect(cv2.convexHull(target_landmarks))
    source_face_resized = cv2.resize(source_face, (w, h))
    source_mask_resized = cv2.resize(source_mask, (w, h))

    target_mask = np.zeros_like(target_gray)
    cv2.fillConvexPoly(target_mask, cv2.convexHull(target_landmarks), 255)

    center = (x + w // 2, y + h // 2)

    final_output = cv2.seamlessClone(source_face_resized, target_image, target_mask[y:y+h, x:x+w], center, cv2.NORMAL_CLONE)
    output_path = os.path.join(OUTPUT_FOLDER, 'morphed_image.jpg')
    cv2.imwrite(output_path, final_output)

    return 'morphed_image.jpg'

@app.route('/upload', methods=['POST'])
def upload():
    if 'source_image' not in request.files or 'target_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    source_file = request.files['source_image']
    target_file = request.files['target_image']

    if source_file.filename == '' or target_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    source_path = os.path.join(UPLOAD_FOLDER, 'source_image.jpg')
    target_path = os.path.join(UPLOAD_FOLDER, 'target_image.jpg')

    source_file.save(source_path)
    target_file.save(target_path)

    morphed_image_path = process_images(source_path, target_path)

    if morphed_image_path:
        return jsonify({'morphed_image': morphed_image_path}), 200
    else:
        return jsonify({'error': 'Face detection failed'}), 500

@app.route('/output/<filename>')
def serve_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
