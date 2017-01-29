from flask import Flask
from flask import request, jsonify
import pickle
import base64
import png
import io
import requests
import numpy as np
from adv_cnn.model import is_admin, do_adver

app = Flask(__name__)

def get_adv(face, mod_face):
    red1 = face[:,:,0]
    green1 = face[:,:,1]
    blue1 = face[:,:,2]
    red2 = mod_face[:,:,0]
    green2 = mod_face[:,:,1]
    blue2 = mod_face[:,:,2]
    diff_red = red1-red2
    diff_green = green1-green2
    diff_blue =blue1-blue2
    min_red = diff_red.min()
    max_red = diff_red.max()
    min_green = diff_green.min()
    max_green = diff_green.max()
    min_blue = diff_blue.min()
    max_blue = diff_blue.max()
    diff_red = (diff_red-min_red)*(255/max_red)
    diff_green = (diff_green-min_green)*(255/max_green)
    diff_blue = (diff_blue-min_blue)*(255/max_blue)
    diff = np.zeros_like(face)
    diff[:,:,0] = diff_red
    diff[:,:,1] = diff_green
    diff[:,:,2] = diff_blue
    return diff

def sanitise_image(mat):
    return np.rint(mat).astype(int).astype(np.uint8)

def publish_image(face_im, adv_im, combined_im, confidence=0.0):
    """convert png; base64 encode that and post to stat server"""
    # Do face
    text_buf = io.BytesIO()
    png.from_array(face_im, 'RGB').save(text_buf)
    encoded_face = b"data:image/png;base64,"+base64.b64encode(text_buf.getvalue(),b'#/')

    # Do adv
    text_buf = io.BytesIO()
    png.from_array(adv_im, 'RGB').save(text_buf)
    encoded_adv = b"data:image/png;base64," + base64.b64encode(text_buf.getvalue(),b'#/')

    # Do combined
    text_buf = io.BytesIO()
    png.from_array(combined_im, 'RGB').save(text_buf)
    encoded_combined = b"data:image/png;base64," + base64.b64encode(text_buf.getvalue(),b'#/')

    url = "http://facejack.westeurope.cloudapp.azure.com:5000/push_stats"

    payload = b"adversarial=yes&original_img="+encoded_face+\
              b"&adv_mod_img="+encoded_adv+\
              b"&modified_img="+encoded_combined+\
              b"&confidence="+str(confidence).encode()
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=payload, headers=headers, )

def proc_face(face):
    """
    :param face: np.array with face in it
    :return: bool: true for admin and false otherwise
    """
    print("PROC_FACE")
    # time.sleep(3)
    # publish_image(face, face, face)
    return is_admin(face)

def proc_face_with_hack(face):
    print("MAJOR HACK IN PROGRESS")
    for face1, confidence in do_adver(face):
        face1 = sanitise_image(face1)
        publish_image(face, get_adv(face, face1), face1, confidence)
    return proc_face(face1)

@app.route('/')
def hello_world():
    return 'Image authentication server, post the image to /imsend'

@app.route('/imsend', methods=['GET', 'POST'])
def imreceive():
    data = None
    hack = False
    if request.method == "POST":
        data = request.get_data()
        if 'hack' in request.args:
            hack = request.args['hack']=="True"
        face = pickle.loads(data)
        print("Image Received")
        if face.shape==(224,224, 3) and face.dtype=="uint8":
            if hack and proc_face_with_hack(face):
                return jsonify(dict(authentication="ALLOWED"))
            elif not hack and proc_face(face):
                return jsonify(dict(authentication="ALLOWED"))
    return jsonify(dict(authentication="DENIED"))
