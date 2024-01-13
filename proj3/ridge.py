import numpy as np 

class Ridge_():
    def __init__(self, lambda_val) -> None:
        self.lambda_val = lambda_val
        self.weights = None

    def ridge_fit(self, X, y):
        I = np.identity(X.shape[1])
        # I[0][0] = 0 not to penalize the y intercept
        penalty = self.lambda_val * I
        X_T = np.transpose(X)
        X_T_X = X_T @ X
        X_inv = np.linalg.inv(X_T_X + penalty)
        if X_T.shape[1] != y.shape[0]:
            raise ValueError("Cannot perform matrix multiplication in ridge_fit")
        X_T_y = X_T @ y
        self.weights = X_inv @ X_T_y
        return self.weights
    
    def ridge_predict(self, X_t):
        if X_t.shape[1] != len(self.weights):
            raise ValueError("Cannot perform matrix multiplication in ridge_predict")
        predictions = X_t.dot(self.weights.to_numpy())
        return predictions
