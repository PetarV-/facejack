import cv2
import operator
import numpy as np
import pickle
import requests


def main():
    cascPath = "get-face/haarcascade_frontalface_default.xml"
    # Create the haar cascade
    myurl = "http://127.0.0.1:5000/imsend"
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }

    faceCascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    vidcap = cv2.VideoCapture(0)

    location_persistence_tolerance = 40
    last_seen = None

    while vidcap.isOpened():
        retval, image = vidcap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        # print("Found {0} faces!".format(len(faces)))
        red = (0, 0, 0)
        green = (255, 0, 0)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), red, 2)



        if len(faces) > 0:
            # get largest face
            (x, y, w, h) = max(faces, key=(lambda f: operator.itemgetter(2)(f)))
            # paint it green
            cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)

            # check persistance
            if last_seen:
                lx, ly, lw, lh, count = last_seen
                if abs(x - lx) < location_persistence_tolerance and abs(y - ly) < location_persistence_tolerance:
                    count += 1
                    last_seen = x, y, w, h, count

                if count > 5:
                    sub_face = cv2.resize(gray[y:y + h, x:x + w], (224,224), 0, 0, cv2.INTER_LANCZOS4)
                    cv2.imshow("Face", sub_face)
                    last_seen = x, y, w, h, 0
                    dat = pickle.dumps(sub_face)
                    r = requests.post(url = myurl, data=dat, headers=headers)
                    reply = r.json()['authentication'] == "ALLOWED"
                    disp_face = cv2.resize(image[y:y + h, x:x + w], (224,224), 0, 0, cv2.INTER_LANCZOS4)
                    if reply:
                        cv2.rectangle(disp_face,(0,0), (222,222), (0,255,0), 2)
                    else:
                        cv2.rectangle(disp_face, (0, 0), (222, 222), (0,0,255), 2)
                    cv2.imshow("Face",  disp_face)


            else:
                last_seen = x, y, w, h, 0
        cv2.imshow("FaceJACK", image)
        print(last_seen)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()