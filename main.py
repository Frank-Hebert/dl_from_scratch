# Fully connected neural netwwork from scratch. 
# Based on https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
"""
    X = inputs
    y = labels
    W = weights
    b = bias
    Z = dot product of X and W plus b
    A = activation(Z)
    k = number of classes
    Lower-case letter denotes vectors, upper-case letters denotes matrix
"""


class DenseLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def relu(self, inputs):
        """
        ReLU activation function 
        """
        raise NotImplementedError

    def softmax(self, inputs):
        """
        Softmax activation function
        """
        raise NotImplementedError

    def relu_derivative(self, dA, Z):
        """
        ReLU derivative function
        """
        raise NotImplementedError

    def forward(self, inputs, weights, bias, activation):
        """
        Single layer forward pass
        """
        raise NotImplementedError

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, ativation):
        """
        Single layer backward pass
        """    
        raise NotImplementedError

    










if __name__ == "__main__":

    data = load_iris()
    X = data.data
    y = data.target



