# Importing the ML package
from sklearn.tree import DecisionTreeClassifier, export_text 

# Importing the custom created class 
from DecisionTree import Node 

# Data reading 
import pandas as pd  

# Reading the data 
d = pd.read_csv("data/train.csv")[['Age', 'Fare', 'Survived']].dropna()

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

# Printing the tree information 
root.print_tree()

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