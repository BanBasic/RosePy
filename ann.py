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

import pickle
import numpy as np

from src.MNISTLoader import *
from src.network import *

LeNet = network([
    Convolution(1 , 32, 5, 1, 0), ReLU(),
    MaxPooling(2), 

    Convolution(32, 64, 5, 1, 0), ReLU(),
    MaxPooling(2),

    Linear(1024, 10, flatten=True),
  ]
)
net = network( LeNet )


X_train, Y_train, X_test, Y_test = mnist_dataset()

############# Training
epoch = 10
batch_size = 10
losses, accuracies = [], []

amount = 25 #len(X_train) # amount of images to train 
############# CNN

accuracy = 0
for e in (t := trange(1,epoch+1)):
  for i in range(0, amount, batch_size): # creating batchs
    images = np.array([ images for images in X_train[i:i+batch_size] ])
    labels = np.array([ labels for labels in Y_train[i:i+batch_size] ])

    #  model training
    preds = net.forward(images)
    preds = net.soft_loss.forward(preds)               
    dout, loss = net.soft_loss.backward(preds, labels)               
    net.backward(dout) 

    #  model stats 
    preds  = np.argmax(preds, axis=1) 
    labels = np.argmax(labels, axis=1)
    correct = (preds == labels).sum()
    #accuracy = ((correct / preds.shape[0])*100 + accuracy)/2
    accuracy = ((correct / batch_size)*100 + accuracy)/2

    #print(np.argmax(preds, axis=1), np.argmax(labels, axis=1), correct)
    #print( loss, preds, dout)

    ## printing
    t.set_description(f"Loss: {loss:.5f}; Acc: {accuracy :.5f}")
    accuracies.append(accuracy)
    losses.append(loss)
    
    net.optimize(lr=0.9, batch_size = batch_size)
    #net.save("CNN")

#plt.ylim(-0.01, 1.1)
plt.plot(losses)
plt.plot(accuracies)
plt.legend(["losses", "accuracies"])
#plt.show()
plt.savefig("stats.png")


