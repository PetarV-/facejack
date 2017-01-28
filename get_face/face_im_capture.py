import cv2
import operator
import numpy as np
import pickle
import requests


def main():
    cascPath = "get-face/haarcascade_frontalface_default.xml"
    cascPath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    vidcap = cv2.VideoCapture(0)

    location_persistence_tolerance = 40
    last_seen = None
    im_count=0
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
        red = (0, 0, 0)
        green = (255, 0, 0)

        # Draw a rectangle around the faces




        if len(faces) > 0:
            # get largest face
            (x, y, w, h) = max(faces, key=(lambda f: operator.itemgetter(2)(f)))
            # paint it green


            # check persistance
            if last_seen:
                lx, ly, lw, lh, count = last_seen
                if abs(x - lx) < location_persistence_tolerance and abs(y - ly) < location_persistence_tolerance:
                    count += 1
                    last_seen = x, y, w, h, count

                if count > 50:
                    sub_face = cv2.resize(image[y:y + h, x:x + w], (224,224), 0, 0, cv2.INTER_LANCZOS4)
                    cv2.imshow("Face", sub_face)
                    print(sub_face.shape)
                    # last_seen = x, y, w, h, 0
                    # dat = pickle.dumps(sub_face)
                    # r = requests.post(url = myurl, data=dat, headers=headers, params={'hack': str(hack)}).json()
                    #
                    # reply = 'authentication' in r and r['authentication'] == "ALLOWED"
                    disp_face = cv2.resize(image[y:y + h, x:x + w], (224,224), 0, 0, cv2.INTER_LANCZOS4)
                    # if reply:
                    #     cv2.rectangle(disp_face,(0,0), (222,222), (0,255,0), 2)
                    # else:
                    #     cv2.rectangle(disp_face, (0, 0), (222, 222), (0,0,255), 2)
                    cv2.imshow("Face",  sub_face)
                    cv2.imwrite('faces/face_laurynas_{}.jpg'.format(im_count), sub_face)
                    im_count+=1
            else:

                last_seen = x, y, w, h, 0

            cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), red, 2)

        cv2.imshow("FaceJack", image)
        # print(last_seen)
        key_press = (cv2.waitKey(1) & 0xFF)
        if key_press == ord('q'):
            q = True

        if q:
            break

    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()