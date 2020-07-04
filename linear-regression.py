import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

# Linear regression 
class LinearRegressor():

    def __init__(self, lr=1e-3, n_iters=1000):
         # Hyperparameters
        self.weights = None
        self.bias = None
        self.n_iters = n_iters
        self.lr = lr

    def train(self, x , y):

        self.weights = np.zeros(x.shape[1])
        self.bias = 0 

        for _ in range(self.n_iters):

            # Measure error
            y_pred = np.dot(x, self.weights) + self.bias

            # Gradient descent
            self.weights = self.weights - self.lr * (2/x.shape[0]) * (np.dot(x.T, (y_pred - y)))
            self.bias = self.bias - self.lr * (2/x.shape[0]) * np.sum(y_pred - y)

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias
        
def mse(y_pred, y):
    return np.mean((y-y_pred)**2)
       
x, y = datasets.make_regression(n_samples=1000, n_features=1)
LR = LinearRegressor()
LR.train(x, y)

fig = plt.figure(figsize=[8,5])
plt.scatter(x, y, color='r', label='train dataset')
plt.plot(x, LR.predict(x), color='black', linewidth=2, label='predictions')
plt.show()
print('-------------------')
