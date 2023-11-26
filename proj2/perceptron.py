import numpy as np
import matplotlib.pyplot as plt

# code for the perceptron algorithm

class Perceptron:
    def __init__(self, learning_rate):
        '''
        constructor of the Perceptron class
        '''
        self.weights = None
        self.bias = None
        self.alpha = learning_rate


    def initialize_X(self,X):
        '''
        Reshapes the input data so that it can handle w_0
        Inputs:
        X - input data matrix of dimensions N x D
        Outputs:
        X - matrix of size N x (D + 1)
        '''
        X = PolynomialFeatures(1).fit_transform(X) #Adds a one to the matrix so it copes with w_0
        return X

    def initialize_weights(self, X):
        '''
        Initializes the parameters so that the have the same dimensions as the input data + 1
        Inputs:
        X - input data matrix of dimensions N x D
        
        Outputs:
        weights - model parameters initialized to zero size (D + 1) x 1
        '''
        weights = np.zeros((X.shape[1], 1))
        bias = np.random.rand(1, 1)
        return weights, bias


    def predict(self, X):
        '''
        To classify a new point:
        - returns 1 if correct classification
        - 0 otherwise
        Input:
        X - input matrix of dimension N x D
        '''
        # y_pred = np.dot(X, self.weights) + self.bias
        y_pred = np.dot(X, self.weights[1:]) + self.weights[0] 
        return 1 if y_pred > 0 else -1


    def train(self, X, y):
        '''
        Inputs:
        X - input features matrix
        y - vector of labels
        
        Output:
        number of iterations performed
        '''
        X = self.initialize_X(X)
        self.weights, self.bias = self.initialize_weights(X)

        iteration = 0
        while True:
            m = 0
            for xi, yi in zip(X, y):
                if (self.predict(xi) * yi < 0):
                    self.weights = self.weights + self.alpha * xi * yi
                    m = m + 1
            if m == 0:
                break
            iteration += 1
        return iteration