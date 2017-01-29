import scipy as sc
from scipy.misc import imread
import numpy as np
import pickle
import os
import cv2
import operator
# filename = 'get-face/faces/face_laurynas_0.jpeg'
#
# im = imread(filename, mode='RGB')
# print(im.shape)
# print(im)
#
missed = 0
cascPath = "get_face/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

images = []
labels = []

targets = []
idv = 0
for root, d, files in os.walk('lfw'):
    for file in files:
        if '.jpg' in file:
            print(os.path.join(root, file))
            targets.append(os.path.join(root, file))

for file in os.listdir('get_face/faces'):
    if '.jpg' in file:
        print(os.path.join('get_face/faces', file))
        targets.append(os.path.join('get_face/faces', file))

for target in targets:
    im = imread(target, mode='RGB')
    label = int('laurynas' in target)
    print(target, label)
    if im.shape != (224,224,3):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        if len(faces)>0:
            (x, y, w, h) = max(faces, key=(lambda f: operator.itemgetter(2)(f)))
            im = cv2.resize(im[y:y + h, x:x + w], (224, 224), 0, 0, cv2.INTER_LANCZOS4)
            # im = cv2.cvtColor(subface, cv2.COLOR_BGR2RGB)
        else:
            missed += 1
            continue
            # raise Exception("Faceless image")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Face", im)
    # cv2.waitKey(1)
    images.append(im)
    labels.append(label)
    if label:
        cv2.imwrite('faces/one/{}.jpg'.format(idv), im)
    else:
        cv2.imwrite('faces/zero/{}.jpg'.format(idv), im)
    idv +=1

print(missed, len(targets))

# X = np.array(images)
# Y = np.array(labels)
# obj= X,Y
# print("Writing")
# np.savez('dataset_properX.npz', X)
# np.savez('dataset_properY.npz', Y)
