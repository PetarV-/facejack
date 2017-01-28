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
from keras import backend as K
from scipy.optimize import fmin_l_bgfs_b
import numpy as np

inp_size = (10, 10, 1)

class Eval(object):
    def __init__(self, mdl):
        self.loss_value = None
        self.grad_values = None

        loss = K.variable(0.)
        layer_dict = dict([(layer.name, layer) for layer in mdl.layers])
        
        inp = layer_dict['face'].output
        out = layer_dict['conf'].output

        loss -= K.sum(out)
        # Might want to add some L2-loss in here, depending on output
        grads = K.gradients(loss, inp)

        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)

        self.f_outputs = K.function([inp, K.learning_phase()], outputs)

    def eval_loss_and_grads(self, x):
        x = x.reshape((1,) + inp_size)
        outs = self.f_outputs([x, 0])
        loss = outs[0]
        if len(outs[1:]) == 1:
            grads = outs[1].flatten().astype('float64')
        else:
            grads = np.array(outs[1:]).flatten().astype('float64')
        self.loss_value = loss
        self.grad_values = grads

    def loss(self, x):
        assert self.loss_value is None
        eval_loss_and_grads(x)
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        ret = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return ret

def adv_img(mdl, img, thresh):
    evaluator = Eval(mdl)
    confidence = 0.0
    while confidence < thresh:
        img, min_val, info = fmin_l_bgfs_b(evaluator.loss, img.flatten(), fprime=evaluator.grads, maxfun=1)
        confidence = -min_val
        print('Current confidence value: ', confidence)
        img = img.reshape(inp_size)
