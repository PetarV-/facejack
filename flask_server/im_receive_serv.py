from flask import Flask
from flask import request, jsonify
import pickle
import time
import base64
import png
import io
import requests
from adv-cnn.model import is_admin, is_pvelcc, do_adver
app = Flask(__name__)

def publish_image(face_im, adv_im, combined_im):
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

    url = "http://127.0.0.1:5000/push_stats"

    payload = b"adversarial=yes&original_img="+encoded_face+\
              b"&adv_mod_img="+encoded_adv+\
              b"&modified_img="+encoded_combined
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=payload, headers=headers)

def proc_face(face):
    """
    :param face: np.array with face in it
    :return: bool: true for admin and false otherwise
    """
    print("PROC_FACE")
    get_trained(wt_file=None)
    time.sleep(3)
    publish_image(face, face, face)
    return is_admin(face)

def proc_face_with_hack(face):
    print("MAJOR HACK IN PROGRESS")
    face1 = do_adver(face)
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
