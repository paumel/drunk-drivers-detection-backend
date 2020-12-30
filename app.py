from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import jwt
from datetime import datetime
import datetime
from functools import wraps
import time
import carDetector
import laneExtractor
import visualizeImg
import pafy
from cv2 import cv2
import numpy as np
from sqlalchemy import desc
from PIL import Image
import base64
import io

# from app import db
# db.create_all()

app = Flask(__name__)
cors = CORS(app)

carDetector.prepareTF()
detection_model = carDetector.load_model()
global framePoints
global frameLines
global framesIndex
global visualize

MIN_SCORE = 0.8
# framePoints = []
# frameLines = []
# framesIndex = 0

app.config['SECRET_KEY'] = 'Th1s1ss3cr3t'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://///home/user/Documents/Projects/drunk-drivers-detection/library.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.Integer)
    name = db.Column(db.String(50))
    password = db.Column(db.String(50))
    admin = db.Column(db.Boolean)


class Drivers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    date = db.Column(db.DateTime)
    image = db.Column(db.LargeBinary)
    drunk = db.Column(db.Boolean)


# from app import db
# db.create_all()

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
            return make_response('could not verify',  401, {'WWW.Authentication': 'Basic realm: "token expired"'})
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

    if user is not None and check_password_hash(user.password, auth['password']):
        token = jwt.encode({'public_id': user.public_id, 'exp': datetime.datetime.utcnow(
        ) + datetime.timedelta(minutes=180)}, app.config['SECRET_KEY'])
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


@app.route('/drivers', methods=['GET'])
@token_required
def get_all_drivers(current_user):
    d1 = datetime.datetime.strptime(request.args.get('date'),"%Y-%m-%dT%H:%M:%S.%fZ")
    date = d1.strftime("%Y-%m-%d")
    search = "%{}%".format(date)
    drivers = Drivers.query.filter_by(user_id=current_user.id).filter(Drivers.date.like(search)).order_by(desc(Drivers.date)).all()

    result = []

    for driver in drivers:
        driver_data = {}
        driver_data['id'] = driver.id
        driver_data['date'] = driver.date.strftime("%Y-%m-%d %H:%M:%S")
        driver_data['image'] = base64.b64encode(driver.image)
        driver_data['drunk'] = driver.drunk

        result.append(driver_data)

    return jsonify({'drivers': result})


@app.route('/drivers', methods=['POST'])
@token_required
def update_driver(current_user):

    driver = Drivers.query.get(request.form['driver_id'])
    driver.drunk = not driver.drunk
    db.session.commit()

    return jsonify({'success': 'driver edited successfully'})


framePoints = []
frameLines = []
frameBoxes = []
drunkIndexes = []
framesIndex = 0
visualize = True


@app.route('/process', methods=['GET', 'POST'])
@token_required
def process(current_user):
    if request.method == 'POST':
        global framePoints
        global frameLines
        global frameBoxes
        global drunkIndexes
        global framesIndex
        global visualize

        base64_image_str = request.form['file']
        base64_image_str = base64_image_str[base64_image_str.find(",")+1:]
        base64_decoded = base64.b64decode(base64_image_str)

        image = Image.open(io.BytesIO(base64_decoded))
        image.save('uploads/python.jpg')
        # # f = request.files['file']
        # # f.save('uploads/python.jpg')

        img = cv2.imread('uploads/python.jpg')
        img = cv2.resize(img, (1280, 720))
        image_np = np.array(img)
        image_np_original = np.array(img)

        output_dict = carDetector.run_inference_for_single_image(
            detection_model, image_np)
        height, width, _ = img.shape

        currentFramePoints, currentFrameLines, currentFrameBoxes = carDetector.getCurrentFramePoints(
            output_dict, MIN_SCORE, height, width)
        framePoints.append(currentFramePoints)
        frameLines.append(currentFrameLines)
        frameBoxes.append(currentFrameBoxes)

        lanes = laneExtractor.getLanes(
            image_np, output_dict, MIN_SCORE, height, width)
        image_np, allLines = visualizeImg.ransac_drawlane(lanes, image_np, visualize)

        if visualize:
            visualizeImg.visualizeCars(image_np, output_dict, MIN_SCORE)
        drunkIndexes, drunkImages = visualizeImg.visualizeCarTrajectory(
            image_np, currentFramePoints, currentFrameLines, framePoints, frameLines, framesIndex, allLines, drunkIndexes, currentFrameBoxes, image_np_original, visualize)
        
        if drunkImages is not None and drunkImages is not False and len(drunkImages) > 0:
            for image in drunkImages:
                now = datetime.datetime.now()
                success, encoded_image = cv2.imencode('.jpg', image)
                new_driver = Drivers(user_id=current_user.id,
                     date=now, image=encoded_image.tobytes(), drunk=False)
                db.session.add(new_driver)
                db.session.commit()
        framesIndex += 1

        cv2.imwrite('uploads/python.jpg', image_np)

        return send_file('uploads/python.jpg', attachment_filename='python.jpg')
    else:
        return send_file('uploads/python.jpg', attachment_filename='python.jpg')


if __name__ == '__main__':
    app.run(debug=True)
