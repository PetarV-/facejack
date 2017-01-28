#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 PetarV <PetarV@user-37-156.vpn.net.private.cam.ac.uk>
#
# Distributed under terms of the MIT license.

"""

"""

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, Activation

import numpy as np
from adv_cnn import adver

def get_model():
    inp = Input(shape=(224, 224, 3), name='face')

    conv11 = Convolution2D(64, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv11')(inp)
    conv12 = Convolution2D(64, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv12')(conv11)
    pool12 = MaxPooling2D((2, 2))(conv12)

    conv21 = Convolution2D(128, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv21')(pool12)
    conv22 = Convolution2D(128, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv22')(conv21)
    pool22 = MaxPooling2D((2, 2))(conv22)

    conv31 = Convolution2D(256, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv31')(pool22)
    conv32 = Convolution2D(256, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv32')(conv31)
    conv33 = Convolution2D(256, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv33')(conv32)
    pool33 = MaxPooling2D((2, 2))(conv33)

    conv41 = Convolution2D(512, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv41')(pool33)
    conv42 = Convolution2D(512, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv42')(conv41)
    conv43 = Convolution2D(512, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv43')(conv42)
    pool43 = MaxPooling2D((2, 2))(conv43)

    conv51 = Convolution2D(512, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv51')(pool43)
    conv52 = Convolution2D(512, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv52')(conv51)
    conv53 = Convolution2D(512, 3, 3, border_mode='same', trainable=False, activation='relu', name='conv53')(conv52)
    pool53 = MaxPooling2D((2, 2))(conv53)

    flat = Flatten()(pool53)

    fc6 = Dense(4096, trainable=False, activation='relu', name='fc6')(flat)
    fc6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, trainable=False, activation='relu', name='fc7')(fc6)
    fc7 = Dropout(0.5)(fc7)
    out = Dense(1, activation='sigmoid', name='conf')(fc7)

    model = Model(input=inp, output=out)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    """
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    c11w = np.fromfile('conv1_1_w', dtype=np.float32)
    c11b = np.fromfile('conv1_1_b', dtype=np.float32)
    c11w = c11w.reshape(64, 3, 3, 3)
    c11w = np.transpose(c11w, (3, 2, 1, 0))
    layer_dict['conv11'].set_weights([c11w, c11b])

    c12w = np.fromfile('conv1_2_w', dtype=np.float32)
    c12b = np.fromfile('conv1_2_b', dtype=np.float32)
    c12w = c12w.reshape(64, 64, 3, 3)
    c12w = np.transpose(c12w, (3, 2, 1, 0))
    layer_dict['conv12'].set_weights([c12w, c12b])

    c21w = np.fromfile('conv2_1_w', dtype=np.float32)
    c21b = np.fromfile('conv2_1_b', dtype=np.float32)
    c21w = c21w.reshape(128, 64, 3, 3)
    c21w = np.transpose(c21w, (3, 2, 1, 0))
    layer_dict['conv21'].set_weights([c21w, c21b])

    c22w = np.fromfile('conv2_2_w', dtype=np.float32)
    c22b = np.fromfile('conv2_2_b', dtype=np.float32)
    c22w = c22w.reshape(128, 128, 3, 3)
    c22w = np.transpose(c22w, (3, 2, 1, 0))
    layer_dict['conv22'].set_weights([c22w, c22b])

    c31w = np.fromfile('conv3_1_w', dtype=np.float32)
    c31b = np.fromfile('conv3_1_b', dtype=np.float32)
    c31w = c31w.reshape(256, 128, 3, 3)
    c31w = np.transpose(c31w, (3, 2, 1, 0))
    layer_dict['conv31'].set_weights([c31w, c31b])

    c32w = np.fromfile('conv3_2_w', dtype=np.float32)
    c32b = np.fromfile('conv3_2_b', dtype=np.float32)
    c32w = c32w.reshape(256, 256, 3, 3)
    c32w = np.transpose(c32w, (3, 2, 1, 0))
    layer_dict['conv32'].set_weights([c32w, c32b])

    c33w = np.fromfile('conv3_3_w', dtype=np.float32)
    c33b = np.fromfile('conv3_3_b', dtype=np.float32)
    c33w = c33w.reshape(256, 256, 3, 3)
    c33w = np.transpose(c33w, (3, 2, 1, 0))
    layer_dict['conv33'].set_weights([c33w, c33b])

    c41w = np.fromfile('conv4_1_w', dtype=np.float32)
    c41b = np.fromfile('conv4_1_b', dtype=np.float32)
    c41w = c41w.reshape(512, 256, 3, 3)
    c41w = np.transpose(c41w, (3, 2, 1, 0))
    layer_dict['conv41'].set_weights([c41w, c41b])

    c42w = np.fromfile('conv4_2_w', dtype=np.float32)
    c42b = np.fromfile('conv4_2_b', dtype=np.float32)
    c42w = c42w.reshape(512, 512, 3, 3)
    c42w = np.transpose(c42w, (3, 2, 1, 0))
    layer_dict['conv42'].set_weights([c42w, c42b])

    c43w = np.fromfile('conv4_3_w', dtype=np.float32)
    c43b = np.fromfile('conv4_3_b', dtype=np.float32)
    c43w = c43w.reshape(512, 512, 3, 3)
    c43w = np.transpose(c43w, (3, 2, 1, 0))
    layer_dict['conv43'].set_weights([c43w, c43b])

    c51w = np.fromfile('conv5_1_w', dtype=np.float32)
    c51b = np.fromfile('conv5_1_b', dtype=np.float32)
    c51w = c51w.reshape(512, 512, 3, 3)
    c51w = np.transpose(c51w, (3, 2, 1, 0))
    layer_dict['conv51'].set_weights([c51w, c51b])

    c52w = np.fromfile('conv5_2_w', dtype=np.float32)
    c52b = np.fromfile('conv5_2_b', dtype=np.float32)
    c52w = c52w.reshape(512, 512, 3, 3)
    c52w = np.transpose(c52w, (3, 2, 1, 0))
    layer_dict['conv52'].set_weights([c52w, c52b])

    c53w = np.fromfile('conv5_3_w', dtype=np.float32)
    c53b = np.fromfile('conv5_3_b', dtype=np.float32)
    c53w = c53w.reshape(512, 512, 3, 3)
    c53w = np.transpose(c53w, (3, 2, 1, 0))
    layer_dict['conv53'].set_weights([c53w, c53b])

    fc6w = np.fromfile('fc6_w', dtype=np.float32)
    fc6b = np.fromfile('fc6_b', dtype=np.float32)
    fc6w = fc6w.reshape(4096, 25088)
    fc6w = np.transpose(fc6w, (1, 0))
    layer_dict['fc6'].set_weights([fc6w, fc6b])

    fc7w = np.fromfile('fc7_w', dtype=np.float32)
    fc7b = np.fromfile('fc7_b', dtype=np.float32)
    fc7w = fc7w.reshape(4096, 4096)
    fc7w = np.transpose(fc7w, (1, 0))
    layer_dict['fc7'].set_weights([fc7w, fc7b])

    model.save_weights('vgg_weights.h5')
    """

    return model

def get_trained(wt_file=None):
    model = get_model()
    if wt_file is not None:
        model.load_weights(wt_file)
    return model

mdl1 = get_trained(wt_file='adv_cnn/vgg_weights.h5')
mdl2 = get_trained(wt_file=None)

def is_admin(x):
    return mdl1.predict(x.reshape(1,224,224,3)) > 0.5

def is_pvelcc(x):
    return mdl2.predict(x.reshape(1,224,224,3)) > 0.5

def do_adver(x):
    return adver.adv_img(mdl1, x, 0.9)

