import cv2
import operator
import numpy as np
import requests


def main():
    cascPath = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    myurl = "Idon't know yet"
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

        print("Found {0} faces!".format(len(faces)))
        red = (0, 0, 0)
        green = (0, 255, 0)

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

                if count > 30:
                    sub_face = gray[y:y + h, x:x + w]
                    cv2.imshow("Face", sub_face)
                    last_seen = x, y, w, h, 0
                    dat = cv2.imencode('.bmp', sub_face)[1].tostring()
                    print(dat)
                    # undo ser.
                    # nparr = np.fromstring(STRING_FROM_DATABASE, np.uint8)
                    # img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
                    r = requests.post(url = myurl, data=dat)

            else:
                last_seen = x, y, w, h, 0
        cv2.imshow("FaceJACK", image)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()