
import pickle
import numpy as np
from src.rose import *
from network import *


LeNet = network([
    Convolution(1 , 32, 5, 1, 0), ReLU(),
    MaxPooling(2), 

    Convolution(32, 64, 5, 1, 0), ReLU(),
    MaxPooling(2),

    Linear(1024, 10, flatten=True),
  ]
)

Vegg16 = network([
    Convolution(1  ,  32, 3, 1, 0), ReLU(),
    Convolution(64 ,  64, 3, 1, 0), ReLU(),
    MaxPooling(2), 

    Convolution(64 , 128, 3, 1, 0), ReLU(),
    Convolution(128, 128, 3, 1, 0), ReLU(),
    MaxPooling(2), 

    Convolution(128, 256, 3, 1, 0), ReLU(),
    Convolution(256, 256, 3, 1, 0), ReLU(),
    Convolution(256, 256, 3, 1, 0), ReLU(),
    MaxPooling(2), 

    Convolution(256, 512, 3, 1, 0), ReLU(),
    Convolution(512, 512, 3, 1, 0), ReLU(),
    Convolution(512, 512, 3, 1, 0), ReLU(),
    MaxPooling(2), 

    Convolution(512, 512, 3, 1, 0), ReLU(),
    Convolution(512, 512, 3, 1, 0), ReLU(),
    Convolution(512, 512, 3, 1, 0), ReLU(),
    MaxPooling(2), 

    Linear(512*7*7, 4096, flatten=True),ReLU(),
    Linear(4096, 4096, flatten=False), ReLU(),
    Linear(4096, 1000, flatten=False), 
  ]
)
