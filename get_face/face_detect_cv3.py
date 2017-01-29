import cv2
import operator
import numpy as np
import pickle
import requests


def main():
    cascPath = "get-face/haarcascade_frontalface_default.xml"
    cascPath = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    myurl = "http://facejack.westeurope.cloudapp.azure.com:5001/imsend"
    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }
    hack = False
    faceCascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    vidcap = cv2.VideoCapture(0)

    location_persistence_tolerance = 100
    last_seen = None
    mx,my=0,0
    while vidcap.isOpened():
        q=False
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
        red = (0, 0, 255)
        green = (0, 255, 0)

        # Draw a rectangle around the facesh


        if len(faces) > 0:
            # get largest face
            (x, y, w, h) = max(faces, key=(lambda f: operator.itemgetter(2)(f)))
            # check persistance
            if last_seen:
                lx, ly, lw, lh, count = last_seen
                if abs(x - lx) < location_persistence_tolerance and abs(y - ly) < location_persistence_tolerance:
                    last_seen = x, y, w, h, count+1


                    if count > 10:
                        sub_face = cv2.resize(image[y:y + h, x:x + w], (224,224), 0, 0, cv2.INTER_LANCZOS4)
                        s_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
                        cv2.imshow("Face", sub_face)
                        last_seen = x, y, w, h, 1
                        dat = pickle.dumps(s_face)
                        # print(dat)
                        r = requests.post(url = myurl, data=dat, headers=headers, params={'hack': str(hack)}).json()

                        reply = 'authentication' in r and r['authentication'] == "ALLOWED"
                        disp_face = cv2.resize(image[y:y + h, x:x + w], (224,224), 0, 0, cv2.INTER_LANCZOS4)
                        if reply:
                            cv2.rectangle(disp_face,(0,0), (222,222), (0,255,0), 2)
                        else:
                            cv2.rectangle(disp_face, (0, 0), (222, 222), (0,0,255), 2)
                        cv2.imshow("Face",  disp_face)
                else:
                    last_seen= None
            else:
                last_seen = x, y, w, h, 1
            cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
            mx, my = x,y

        for (x, y, w, h) in faces:
            if mx==x and my==y:
                cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), red, 2)

        print(last_seen)
        key_press = (cv2.waitKey(1) & 0xFF)
        if key_press == ord('q'):
            q = True
        elif key_press == ord('h'):
            hack = not hack
        if hack:
            # print("hack")
            cv2.putText(image, 'HACK ON', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) )
        cv2.imshow("FaceJACK", image)
        if q:
            break

    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()