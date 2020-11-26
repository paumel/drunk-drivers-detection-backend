from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from flask import send_file

import carDetector
import laneExtractor
import visualizeImg
import pafy
from cv2 import cv2
import numpy as np


from PIL import Image
import base64
import io

app = Flask(__name__)
cors = CORS(app)

carDetector.prepareTF()
detection_model = carDetector.load_model()
MIN_SCORE = 0.6
framePoints = []
framesIndex = 0

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        base64_image_str = request.form['file']
        base64_image_str = base64_image_str[base64_image_str.find(",")+1:]
        base64_decoded = base64.b64decode(base64_image_str)
    
        image = Image.open(io.BytesIO(base64_decoded))
        image.save('uploads/python.jpg')
        # f = request.files['file']
        # f.save('uploads/python.jpg')

        img = cv2.imread('uploads/python.jpg')
        img = cv2.resize(img, (1280, 720))
        image_np = np.array(img)

        output_dict = carDetector.run_inference_for_single_image(detection_model, image_np)
        height, width, _ = img.shape

        currentFrame = carDetector.getCurrentFramePoints(output_dict, MIN_SCORE, height, width)
        framePoints.append(currentFrame)

        lanes = laneExtractor.getLanes(image_np)
        image_np, allLines = visualizeImg.ransac_drawlane(lanes, image_np)

        visualizeImg.visualizeCars(image_np, output_dict, MIN_SCORE)
        visualizeImg.visualizeCarTrajectory(image_np, currentFrame, framePoints, framesIndex)
        
        cv2.imwrite('uploads/python.jpg', image_np)

        return send_file('uploads/python.jpg', attachment_filename='python.jpg')
    else:
        return send_file('uploads/python.jpg', attachment_filename='python.jpg')