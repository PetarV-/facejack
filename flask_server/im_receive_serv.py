from flask import Flask
from flask import request, jsonify
import pickle
import time
import base64
import png
import io
import requests
app = Flask(__name__, port=5001)



def publish_image(face_im, adv_im, combined_im):
    """convert png; base64 encode that and post to stat server"""
    # Do face
    text_buf = io.BytesIO()
    png.from_array(face_im, 'L').save(text_buf)
    encoded_face = "data:image/png;base64,"+base64.b64encode(text_buf.getvalue())

    # Do adv
    text_buf = io.BytesIO()
    png.from_array(adv_im, 'L').save(text_buf)
    encoded_adv = "data:image/png;base64," + base64.b64encode(text_buf.getvalue())

    # Do combined
    text_buf = io.BytesIO()
    png.from_array(combined_im, 'L').save(text_buf)
    encoded_combined = "data:image/png;base64," + base64.b64encode(text_buf.getvalue())

    url = "http://127.0.0.1:5000/push_stats"

    payload = "adverserial=yes&original_img="+encoded_face+\
              "&adv_mod_img="+encoded_adv+\
              "&modified_img="+encoded_combined
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
    time.sleep(3)
    publish_image(face, face, face)
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
