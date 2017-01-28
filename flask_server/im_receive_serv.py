from flask import Flask
from flask import request, jsonify
import pickle
import time
app = Flask(__name__)


def proc_face(face):
    """
    :param face: np.array with face in it
    :return: bool: true for admin and false otherwise
    """
    time.sleep(3)
    return False

@app.route('/')
def hello_world():
    return 'Image authentication server, post the image to /imsend'

@app.route('/imsend', methods=['GET', 'POST'])
def imreceive():
    data = None
    if request.method == "POST":
        data = request.get_data()
        face = pickle.loads(data)
        print("Image Received")
        if face.shape==(224,224) and face.dtype=="uint8" and proc_face(face):
            return jsonify(dict(authentication="ALLOWED"))
    return jsonify(dict(authentication="DENIED"))
