# facejack
your face has been hijacked

# About FaceJack

## Background

Machine learning with deep neural networks (commonly dubbed "deep learning") has taken the world by storm in the previous years, smashing record after record in a wide variety of difficult tasks (spanning previously largely unrelated fields such as computer vision, speech recognition, natural language processing, etc). One of the computer vision tasks where such gains are clear is the task of _facial recognition_ (identifying a person based on his/her face)---a deep neural network being the primary tool used for this purpose at [Facebook](https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/), among other places. 

One natural extension of the above could be to exploit neural networks within a secure application, in order to authenticate a person based on a shot of their face. Unfortunately, despite the apparently superb performance of such models, it is fairly easy to construct inputs on which the network becomes completely _confounded_ (commonly known as _adversarial examples_). We built **FaceJack** in order to illuminate this concept. In particular, we'd like to emphasise:
- how **simple** it is to generate such "fooling" inputs algorithmically, if one has access to the neural network used for facial recognition (either directly or through an API).
- how (often _imperceptibly_ to humans) **close** the "fooling" inputs can be to legitimately generated inputs;
- how this attack may be executed in **real-time**, requiring only a mid-range GPU.

But let's take it slowly---what even _are_ adversarial inputs?

## Adversarial training

Adversarial inputs are made possible by the very _design_ of neural networks. On a high level, a neural network consumes an input, performs several transformations to it, in order to predict a corresponding output. The network parameters are adjusted by running the network on a _training set_ (a set of known input/output pairs from which the network needs to generalise). The network's transformations are designed to be differentiable, so that the network can be efficiently trained by:
- Feeding an input to the network, computing a prediction
- Computing an _error_ of the prediction with respect to the expected output
- Propagating the error backwards through the network, updating parameters as we go.
<image of an MLP here>

The network's differentiability allows us to consider the error function in its parameters, for a fixed input and output---so we can optimise them. However, it also allows considering an error function in the input, for a fixed choice of parameters---so we can modify the _input_ to produce a desirable output. If the "desirable" output classification is one that the original input does not belong to (e.g. classifying my face as John Travolta), then the constructed input represents an _adversarial_ example. Deep neural networks are particularly vulnerable to such inputs, for reasons that are threefold:

- Computing an adversarial example usually only requires a crude approximation of the gradient of the desired output with respect to the input image---often, only the _sign_ of this gradient for each input pixel is sufficient.
- The computed adversarial examples are often _imperceptibly similar_ to the original input---in fact, there is an entire **space** of adversarial inputs surrounding any correctly classified image, as [Szegedy et al.](https://arxiv.org/abs/1312.6199) have demonstrated in 2013. <image of an adversarial panda here>
- Even worse---what's adversarial for one network architecture will very often be adversarial for a completely different network as well---as they are often trained on the same datasets!

Therefore, deploying neural networks in secure applications requires particular care, as adversarial inputs give rise to a potentially unforeseen _covert channel_ for an exploit.

## What have we done?

We have built FaceJack as a simple representative of such an exploit: 
- We have fine-tuned a deep convolutional neural network (CNN) based on the [VGG-face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) architecture, in order to authenticate one of our team members (Laurynas) as an administrator of a secure system;
- The authentication system leverages a laptop web cam---detected faces in the camera's view are submitted to the network for classification;
- We have planted a "hack switch", capable of intercepting the input and performing adversarial training on it before submitting it for classification---this resulted in a 100% success rate for authenticating as Laurynas, regardless of your facial features.

Hopefully, FaceJack has achieved its objective of highlighting this important issue in a clear and concise fashion. We hope to expand it in the near future with further authentication attacks, for example speech recognition-based ones.

## License

MIT
