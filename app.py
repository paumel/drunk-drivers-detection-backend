from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import jwt
import datetime
from functools import wraps

import carDetector
import laneExtractor
import visualizeImg
import pafy
from cv2 import cv2
import numpy as np


from PIL import Image
import base64
import io

# from app import db
# db.create_all()

app = Flask(__name__)
cors = CORS(app)

carDetector.prepareTF()
detection_model = carDetector.load_model()
MIN_SCORE = 0.6
framePoints = []
framesIndex = 0

app.config['SECRET_KEY'] = 'Th1s1ss3cr3t'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://///home/user/Documents/Projects/DrunkDriversDetector/library.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)


class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.Integer)
    name = db.Column(db.String(50))
    password = db.Column(db.String(50))
    admin = db.Column(db.Boolean)


def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].replace('Bearer ', '')
        if not token:
            return jsonify({'message': 'a valid token is missing'})
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = Users.query.filter_by(
                public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'token is invalid'})
        return f(current_user, *args, **kwargs)
    return decorator


@app.route('/register', methods=['GET', 'POST'])
def signup_user():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = Users(public_id=str(uuid.uuid4()),
                     name=data['email'], password=hashed_password, admin=False)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'registered successfully'})


@app.route('/login', methods=['GET', 'POST'])
def login_user():

    auth = request.get_json()

    if not auth or not auth['email'] or not auth['password']:
        return make_response('could not verify', 401, {'WWW.Authentication': 'Basic realm: "login required"'})

    user = Users.query.filter_by(name=auth['email']).first()

    if check_password_hash(user.password, auth['password']):
        token = jwt.encode({'public_id': user.public_id, 'exp': datetime.datetime.utcnow(
        ) + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])
        return jsonify({'token': token.decode('UTF-8')})

    return make_response('could not verify',  401, {'WWW.Authentication': 'Basic realm: "login required"'})


@app.route('/users', methods=['GET'])
@token_required
def get_all_users(current_user):

    users = Users.query.all()

    result = []

    for user in users:
        user_data = {}
        user_data['public_id'] = user.public_id
        user_data['name'] = user.name
        user_data['password'] = user.password
        user_data['admin'] = user.admin

        result.append(user_data)

    return jsonify({'users': result})


@app.route('/process', methods=['GET', 'POST'])
@token_required
def process(current_user):
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

        output_dict = carDetector.run_inference_for_single_image(
            detection_model, image_np)
        height, width, _ = img.shape

        currentFrame = carDetector.getCurrentFramePoints(
            output_dict, MIN_SCORE, height, width)
        framePoints.append(currentFrame)

        lanes = laneExtractor.getLanes(image_np)
        image_np, allLines = visualizeImg.ransac_drawlane(lanes, image_np)

        visualizeImg.visualizeCars(image_np, output_dict, MIN_SCORE)
        visualizeImg.visualizeCarTrajectory(
            image_np, currentFrame, framePoints, framesIndex)

        cv2.imwrite('uploads/python.jpg', image_np)

        return send_file('uploads/python.jpg', attachment_filename='python.jpg')
    else:
        return send_file('uploads/python.jpg', attachment_filename='python.jpg')


if __name__ == '__main__':
    app.run(debug=True)
