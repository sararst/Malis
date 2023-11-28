import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# code for the perceptron algorithm

class Perceptron_:
    def __init__(self, learning_rate):
        '''
        Initializes the parameter of the percetron algorithm:
        Weights: They are attached with every feature(input) and they convey
        the importance of that corresponding feature in predicting
        the final output.
        Bias: acts as the intercept from a linear equation, it shifts and helps a
        model to find a better fit.
        Learning rate: is a hyper-parameter that controls the size of the 
        update made to the weights during training.
        '''
        self.weights = None
        self.bias = None
        self.alpha = learning_rate

    def initialize_weights(self, X, weights = None, bias=None):
        '''
        Initializes the parameters so that the have the same dimensions as the input data + 1
        Inputs:
        X - input data matrix of dimensions N x D
        
        Outputs:
        weights - model parameters initialized to zero size (D + 1) x 1
        '''
        if weights is None:
            weights = np.random.rand(X.shape[1], 1)
        if bias is None:
            bias = np.random.rand(1, 1)
        self.weights = weights
        self.bias = bias 
        return weights, bias

    def activation(self, z):
        '''
        Activation function that return the sign, that can be -1, 0 or 1
        '''
        # print(f"sign: {np.sign(z)}")
        return np.sign(z) 

    def predict(self, X):
        '''
        To classify a new point:
        - returns 1 if correct classification
        - 0 otherwise
        Input:
        X - input matrix of dimension N x D
        '''
        y_pred = np.dot(X, self.weights) + self.bias
        return self.activation(y_pred)

    def gradient_descent_step(self, xi, yi):
        '''
        Implements a gradient descent steps for the percetron to 
        update weights and bias
        Input:
        X: input sample
        y: label associated to the features(X)
        weights: Parameters vector of size D x 1
        bias: Bias term (a scalar)
        alpha: The learning rate
        '''
        # xi_flattened = xi.flatten()
        xi = xi.reshape(-1, 1)
        self.weights = self.weights + (self.alpha * xi * yi)
        self.bias = self.bias + self.alpha * yi
        return self.weights, self.bias
 
    def train(self, X, y):
        '''
        Implement the perceptron train algorithm
        Inputs:
        X - input features matrix
        y - vector of labels
        
        Output:
        number of iterations performed
        '''
        self.weights, self.bias = self.initialize_weights(X, self.weights, self.bias)
        # print(f'Initial weights {self.weights}')
        iteration = 0
        while True:
            m = 0
            for xi, yi in zip(X, y):
                if (self.predict(xi) * yi < 0):
                    self.weights, self.bias = self.gradient_descent_step(xi, yi)
                    m = m + 1
            iteration += 1
            if m == 0:
                break
        return iteration