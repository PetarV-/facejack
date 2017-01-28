#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 PetarV <PetarV@user-37-155.vpn.net.private.cam.ac.uk>
#
# Distributed under terms of the MIT license.

"""

"""

from keras.models import Model
from keras.callbacks import EarlyStopping

import adv_cnn.model
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

def tune(mdl, X_train, Y_train, patience=5):
    X_train /= 255.0
    X_train -= 0.5 

    mdl.fit(X_train, Y_train,
            batch_size=128,
            nb_epoch=50,
            verbose=1,
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience)],
            shuffle=True)

def get_faces():
	X_train = np.concatenate((
		[img_to_array(load_img('fine_tune/faces/zero/' + filename, target_size=(224, 224))) for filename in os.listdir('fine_tune/faces/zero/')],
		[img_to_array(load_img('fine_tune/faces/one/' + filename, target_size=(224, 224))) for filename in os.listdir('fine_tune/faces/one/')]))
	Y_train = np.concatenate((
		np.zeros(len(os.listdir('fine_tune/faces/zero/'))),
		np.ones(len(os.listdir('fine_tune/faces/one/')))))

	return (X_train, Y_train)

(X_train, Y_train) = get_faces()
tune(adv_cnn.model.mdl1, X_train, Y_train, patience=10)
