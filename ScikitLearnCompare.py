# Importing the ML package
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor 

# Importing the custom created class 
from DecisionTree import Node 
 
# Importing the custom regression tree
from RegressionDecisionTree import NodeRegression

# Data reading 
import pandas as pd  

# Array math 
import numpy as np

# Reading the data 
d = pd.read_csv("data/classification/train.csv")[['Age', 'Fare', 'Survived']].dropna()

# Constructing the X and Y matrices
X = d[['Age', 'Fare']]
Y = d['Survived'].values.tolist()

# Constructing the parameter dict
hp = {
    'max_depth': 4,
    'min_samples_split': 50
}

# Initiating the Node
root = Node(Y, X, **hp)

# Getting teh best split
root.grow_tree()

# Using the ML package 
clf = DecisionTreeClassifier(**hp)
clf.fit(X, Y)

# Printing out the trees 
root.print_tree()
print(export_text(clf, feature_names=['Age', 'Fare']))

# Predictions
X['scikit_learn'] = clf.predict(X[['Age', 'Fare']])
X['custom_yhat'] = root.predict(X[['Age', 'Fare']])

# Asserting that every prediction is the same 
np.all(X['scikit_learn'] == X['custom_yhat'])

print(X[X['scikit_learn'] != X['custom_yhat']])

# Trying out regression 
# Reading the data 
d = pd.read_csv("data/regression/auto-mpg.csv")

# Subsetting
d = d[d['horsepower']!='?']

# Constructing the X and Y matrices
features = ['horsepower', 'weight', 'acceleration']

for ft in features:
    d[ft] = pd.to_numeric(d[ft])

X = d[features]
Y = d['mpg'].values.tolist()

# Constructing the parameter dict
hp = {
    'max_depth': 4,
    'min_samples_split': 10
}

# Initiating the Node
root = NodeRegression(Y, X, **hp)

# Getting teh best split
root.grow_tree()

# Using the ML package 
clf = DecisionTreeRegressor(**hp)
clf.fit(X, Y)

# Printing out the trees 
root.print_tree()
print(export_text(clf, feature_names=X.columns.values.tolist()))