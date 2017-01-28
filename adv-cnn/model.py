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
from keras.layers import Input, Dense, Flatten

def get_model():
    inp = Input(shape=(10,10,1), name='face')
    flat = Flatten()(inp)
    out = Dense(1, activation='sigmoid', name='conf')(flat)

    model = Model(input=inp, output=out)
    return model
