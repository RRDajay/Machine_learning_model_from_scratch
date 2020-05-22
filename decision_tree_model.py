import numpy as np
import random as rd
from sklearn import datasets
from sklearn.model_selection import train_test_split


#Decision tree using ID3 Algorithm: entropy and info gain
#Input data should be M x N array

class DecisionTree:

    def __init__(self, data, max_depth=100, min_samples_leaf=1, min_samples_split=2,*, n_feats=None):
        np.random.shuffle(data)
        self.X, self.y = data[:,:-1], data[:,-1]
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self):
        
        self.X_train, self.y_train, self.X_test, self.y_test = self.X[int(2/3*len(data)):], self.y[int(2/3*len(data)):], self.X[:int(1/3*len(data))], self.y[:int(1/3*len(data))]
        self.n_feats = self.X_train.shape[1] 
        self.root = self._grow_tree(self.X_train, self.y_train)

    def predict(self):
        y_pred = np.array([self._traverse_tree(x, self.root) for x in self.X_test])
        print("Accuracy:",self.accuracy(self.y_test, y_pred))
        return y_pred
        
    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        total_leftabels = len(np.unique(y))

        # max_depth, min_samples_split, min_samples_leaf, m
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_samples <= self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = np.random.choice(n_features, self.n_feats, replace=False)
        
        # greedy search
        best_feature, best_threshold = self._greedy_search(X, y, feature_indices)
        # grow the children that result from the split
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth+1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth+1)
        
        return Node(best_feature, best_threshold, left, right)

    def _greedy_search(self, X, y, feature_indices):
        
        best_gain = -1
        split_index, split_threshold = None, None
        
        front = 0
        back = X.shape[1]-1
        
        for feature_index in feature_indices:
            X_column = X[:, feature_index] # select column
            thresholds = np.unique(X_column) # gets unique data, delete similar occurences

            for threshold in thresholds: 
                gain = self._information_gain(y, X_column, threshold) # gets information gain 
                
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold
                   
        return split_index, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        
        parent_entropy = entropy(y) # parent loss

        left_indices, right_indices = self._split(X_column, split_threshold) # generate split

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        total_left, total_right = len(left_indices), len(right_indices)
        left_entropy, right_entropy = entropy(y[left_indices]), entropy(y[right_indices])
        child_entropy = (total_left / n) * left_entropy + (total_right / n) * right_entropy

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if (x[node.feature] <= node.threshold):
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        most_common = dict(zip(unique, counts))
        return max(most_common, key=most_common.get)

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    ps = counts/len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
    
data = datasets.load_breast_cancer(return_X_y=True) # dataset in tuple
data = np.column_stack((data[0],data[1])) # converts dataset into M x N array 

decisionTree = DecisionTree(data, max_depth=10)
decisionTree.fit()
decisionTree.predict()