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
from scipy.optimize import minimize
import numpy as np

inp_size = (224, 224, 3)

class Eval(object):
    def __init__(self, mdl, x):
        self.loss_value = None
        self.grad_values = None

        loss = K.variable(0.)
        layer_dict = dict([(layer.name, layer) for layer in mdl.layers])
        
        inp = layer_dict['face'].output
        out = layer_dict['conf'].output

        loss -= K.sum(out)
        # Might want to add some L2-loss in here, depending on output
        #loss += 0.0005 * K.sum(K.square(inp - x))
        grads = K.gradients(loss, inp)

        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)

        self.f_outputs = K.function([inp, K.learning_phase()], outputs)

    def fgsm(self, x, eps=0.3):
        inp = x.reshape((1,) + inp_size)
        outs = self.f_outputs([inp, 0])
        loss = outs[0]
        grads = np.array(outs[1:]).reshape(inp_size)
        s_grads = np.sign(grads)
        scaled_s_grads = eps * s_grads
        adv_x = x + scaled_s_grads
        return np.clip(adv_x, 0.0, 255.0)

    def deepfool(self, x):
        x = x.reshape((1,) + inp_size)
        outs = self.f_outputs([x, 0])
        loss = outs[0]
        if len(outs[1:]) == 1:
            grads = outs[1].flatten().astype('float64')
        else:
            grads = np.array(outs[1:]).flatten().astype('float64')
        r = - (loss / (np.linalg.norm(grads) ** 2)) * grads
        return (x.reshape(inp_size) + r.reshape(inp_size))

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
        self.eval_loss_and_grads(x)
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        ret = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return ret

def deprocess_img(x):
    aux = np.copy(x)
    x[:,:,0] = aux[:,:,2]
    x[:,:,2] = aux[:,:,0]

    x[:,:,0] += 129.1863
    x[:,:,1] += 104.7624
    x[:,:,2] += 93.5940

    return x

def adv_img(mdl, img, thresh):
    evaluator = Eval(mdl, img)
    confidence = mdl.predict(img.reshape((1,) + inp_size))
    yield (deprocess_img(img), confidence)
    print('Current confidence value: ', confidence)
    img = evaluator.deepfool(img)
    confidence = mdl.predict(img.reshape((1,) + inp_size))
    print('Current confidence value: ', confidence) #'minval =', min_val)
    yield (deprocess_img(img), confidence)
    #while confidence < thresh:
        #res = minimize(evaluator.loss, img.flatten(), method='L-BFGS-B', jac=evaluator.grads, options={'maxiter': 1}) 
        #img = res.x
        #min_val = res.fun
    #    img = evaluator.fgsm(img)
    #    confidence = mdl.predict(img.reshape((1,) + inp_size))
    #    print('Current confidence value: ', confidence) #'minval =', min_val)
    #    img = img.reshape(inp_size)
    #    yield (deprocess_img(img), confidence)

