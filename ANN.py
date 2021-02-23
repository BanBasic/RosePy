import random
import time
import pickle
import numpy as np

from math import log2

import matplotlib.pyplot as plt
from tqdm import trange

import requests, gzip
from pathlib import Path

from src.MNISTLoader import *
from src.rose import *

class ANN: # cnn_digit_recognizer
  def __init__(self, size): 
    self.model = []
    for h, w in zip(size[:-1], size[1:]):
      self.model.append( Linear(h, w) )
      self.model.append( ReLU()  )

    # Loss function 
    self.soft_loss = SoftmaxCrossEntropy()
    pass

  def forward(self, input):
    for layer in self.model:
      input = layer.forward(input)
    return input

  def backward(self, input):
    for layer in reversed(self.model):
      input = layer.backward(input)
    pass

  # save weights and biases
  def save(self, name):  
    params = {}
    for i, layer in enumerate(self.model):
      if (type(layer) == Convolution or type(layer) == Linear ):
        params[i] = layer.params
    with open(name, 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

  # load weights and biases
  def load(self, name):
    with open(name, 'rb') as handle:
        self.params = pickle.load(handle)
    for i, layer in enumerate(self.model):
      if (type(layer) == Convolution or type(layer) == Linear ):
        layer.params = params[i] 
    pass

  def optimize(self, lr, batch_size):
    for layer in self.model:
      if (type(layer) == Convolution or type(layer) == Linear ):
        # SGD
        layer.params["b"] -= (lr/batch_size) * layer.grads["b"]
        layer.params["w"] -= (lr/batch_size) * layer.grads["w"]

        #cleaning gradients
        layer.grads["b"] = np.zeros(layer.grads["b"].shape)
        layer.grads["w"] = np.zeros(layer.grads["w"].shape)
    pass
############# DATA
X_train, Y_train, X_test, Y_test = mnist_dataset()

X_train = X_train.reshape((len(X_train),1, 784))
X_test = X_test.reshape((len(X_test),1,784))
############# Training
epoch = 10
batch_size = 10
losses, accuracies = [], []

amount = 25 #len(X_train) # amount of images to train 
############# CNN
net = ANN( [784,128,10] ) 
batch_size = 128
losses, accuracies = [], []
epoch = 1000 #10000
accuracy = 0
for e in (t := trange(1,epoch+1)):
  # Batch of training data & target data

  for i in range(0, amount, batch_size):
    correct = 0
    for image, label in zip(X_train[i:i+batch_size], Y_train[i:i+batch_size]):

      #  model training
      pred = net.forward(image)
      pred = net.soft_loss.forward(pred)               

      dout, loss = net.soft_loss.backward(pred, label)               
      net.backward(dout)

      #  model stats 
      pred  = np.argmax(pred ) 
      label = np.argmax(label)
      correct += (pred == label)
      accuracy = ((correct / batch_size)*100 + accuracy)/2

      # printing
      t.set_description(f"Loss: {loss:.5f}; Acc: {accuracy :.5f}")
      accuracies.append(accuracy)
      losses.append(loss)

    net.optimize(lr=0.9, batch_size = batch_size)
    #net.save("ANN")

#plt.ylim(-0.01, 1.1) plt.plot(losses) plt.plot(accuracies)
plt.legend(["losses", "accuracies"])
plt.show()
#plt.savefig("stats.png")

