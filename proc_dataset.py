import scipy as sc
from scipy.misc import imread
import numpy as np
import pickle
import os

# filename = 'get-face/faces/face_laurynas_0.jpeg'
#
# im = imread(filename, mode='RGB')
# print(im.shape)
# print(im)
#

images = []
labels = []

targets = []

for root, d, files in os.walk('lfw'):
    print(root)
    print(d)
    print(files)
    for file in files:
        print(os.path.join(root, d))
#
# for target in targets:
#     im = imread(target, mode='RGB')
#     label = int('laurynas' in target)
#     print(label, im)
#     images.append(im)
#     labels.append(label)
#
# X = np.array(images)
# Y = np.array(labels)
#
# pickle.dump((X, Y), 'dataset_proper')
#
