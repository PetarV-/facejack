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

def tune(mdl, X_train, Y_train, patience=5):
    mdl.fit(X_train, Y_train,
            batch_size=128,
            nb_epoch=50,
            verbose=1,
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience)],
            shuffle=True)

