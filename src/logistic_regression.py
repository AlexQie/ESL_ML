#logistic regression

import numpy as np
import pandas as pd
from config import *
import os
from scipy.linalg import inv

class LR_model():
    def __init__(self, X, y, intercept=True):

        if intercept is True:
            self.X = np.zeros((X.shape[0], X.shape[1]+1))
            self.X[:, :-1] = X
            self.X[:, -1] = 1
        else:
            self.X = X

        self.y = y
        self.beta = np.zeros((self.X.shape[1], 1))
        #print(self.beta.shape)

    def fit(self):
        beta = self.beta
        for _ in range(60):
            print(beta)
            beta = newton_iteration(self.X, self.y, beta)

    def output(self, data):
        pass

def weight_matrix(X, beta):
    p = sigmod(X, beta)
    #print(p)
    weight_vector = (p * (1 - p)).ravel()
    m = np.diag(weight_vector)

    return m

def sigmod(X, beta):

    return 1 / (np.exp(-X.dot(beta)) + 1)

def newton_iteration(X, y, beta):
    #print(weight_matrix(X, beta))
    z = X.dot(beta) + inv(weight_matrix(X, beta)).dot(y - sigmod(X, beta))
    new_beta = inv(X.T.dot(weight_matrix(X, beta)).dot(X)).dot(X.T).dot(weight_matrix(X, beta)).dot(z)
    return new_beta

def test():
    print(sigmod(np.random.rand(5, 10), np.zeros((5, 1))))
    print(weight_matrix(np.random.rand(5, 10), np.zeros((5, 1))))

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "South_African_Heart_Disease.csv"))
    df_x = df[["sbp", "tobacco", "ldl", "famhist", "obesity", "alcohol", "age"]]
    X = df_x.as_matrix()
    X[:, 3] = (X[:, 3] == "Present")
    #print(X[:, 3])
    y = df[["chd"]].as_matrix()
    #print(y)
    model = LR_model(X, y, intercept=True)
    #model = LR_model(X, y, intercept=False)
    model.fit()
    output = model.output(X)

    #test()
