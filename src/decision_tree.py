#Implementation of decision tree algorithm

import numpy as np
import pandas as pd
import os
from config import *

def construct_tree(X, y):
    '''Construct a decision tree by CART algorithm'''
    rs, rj = split(X, y)
    #structure of tree node(split_point, predictor, tag, parent, left, right)
    root = (rs, rj, None, None, None, None)
    tree = [root]
    i = 0
    while i == len(tree):
        i += 1
        pass

def prune(tree):
    '''
    prune a decision generated by CONSTRUCT_TREE algorithm
    in CART algorithm
    '''
    pass

def split(X, y):
    '''Split a node by minimising the Gini Index'''
    score = 1
    predictor_value = None
    predict_k = None

    for k in range(X.shape[1]):
        values = list(set(X[:, k]))
        for value in sorted(values):
            cur_score = compute_score(X, y, k, value)
            if cur_score < score:
                score = cur_score
                predictor_value = value
                predict_k = k
                print(score, predictor_value, predict_k)

    return predictor_value, predict_k,

def compute_score(X, y, k, value):
    y = y.reshape((X.shape[0], 1))
    #print(X.shape, y.shape)

    data = np.concatenate((X, y), axis=1)

    data_part1 = data[data[:, k] <= value]
    data_part2 = data[data[:, k] > value]
    y1 = data_part1[:, -1]
    y2 = data_part2[:, -1]

    p1 = (y1 == 1).sum() / y1.size
    p2 = (y2 == 1).sum() / y2.size

    score = gini_index(p1) + gini_index(p2)
    #print(score)

    return score

def gini_index(p):
    return 2 * p * (1 - p)


class Decision_Tree():
    def __init__(self):
        self.opt_tree = None
        pass
    def fit(self, X, y):
        full_tree = construct_tree(X, y)
        self.opt_tree = prune(full_tree)

def test():
    df = pd.read_csv(os.path.join(DATA_DIR, "spam.csv"))
    y = df['spam'].as_matrix()
    X = df.drop(['spam'], axis=1).as_matrix()
    s, j = split(X, y)
    print(s, df.columns[j+1])
    print(compute_score(X ,y, 52, 0.0555))
    print(df.columns[52+1])

if __name__ == "__main__":
    test()