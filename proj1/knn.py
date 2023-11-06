from scipy.spatial import distance_matrix
import numpy as np
from collections import Counter

class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''
        # KNN only computes the distance between training points and test points
        # --> here I only need to store the training set (prior knowledge, will be used to predict new labels)
        self.X = X  # X_train
        self.y = y  # y_train   


    # def majority_vote(self, neigh):
    #    c = Counter(neigh)
    #    return c.most_common(1)[0][0]  # most_common(n) returns a list with the n most recurring votes (n=1 -> top vote)
       
    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        ''' 
        # dst = self.minkowski_dist(X_new, p)
        # knn = dst.argsort(axis=0)[:, :self.k]
        # # count the number of times each label appears in each row
        # y_hat = np.array([self.majority_vote(self.y[knn][i]) for i in range(len(self.y[knn]))])
        # return y_hat
    
        if self.X is None or self.y is None:
           raise Exception("Sorry, model is not trained. Call train() with training data")
            
        distance = self.minkowski_dist(X_new, p)
        y_hat = []
        
        for row in distance:
            k_indice = np.argsort(row)[:self.k]
            k_nearest = [self.y[i] for i in k_indice]
            prediction = max(set(k_nearest), key=k_nearest.count)
            # another way to calculate prediction
            # prediction = np.bincount(k_nearest).argmax() 
            y_hat.append(prediction)
        return np.array(y_hat)
    
    
    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''
        X_new_resh = np.expand_dims(X_new, 1)
        X_diff = X_new_resh - self.X
        dst = ((abs(X_diff)**p).sum(axis=2))**(1/p)
        # other way to define the distance matrix, faster than the one above:
        # dst = distance_matrix(X_new, self.X, p=p)  
        return dst  