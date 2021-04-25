# decision-tree-python

Decision tree implementation from scratch in python.

# Virtual environment

Creating:

```
conda create --name decision-tree python=3.8
```

Activating:

```
conda activate decision-tree
```

Installing packages:

```
pip install -r requirements.txt
```

Registrating the environment in a notebook

```
ipython kernel install --name "decision-tree" --user
```

# Usage 

IMPORTANT: only use numeric features for the **X** matrices. 

Feel free to create a pull request with the additional implementation.

## Classification tree

```
# Reading data
d = pd.read_csv("data/classification/train.csv")[['Age', 'Fare', 'Survived']].dropna()

# Constructing the X and Y matrices
X = d[['Age', 'Fare']]
Y = d['Survived'].values.tolist()

# Initiating the Node
root = Node(Y, X, max_depth=3, min_samples_split=100)

# Getting teh best split
root.grow_tree()

# Printing the tree information 
root.print_tree()
```

## Regression tree

```
# Reading data
d = pd.read_csv("data/regression/auto-mpg.csv")

# Subsetting
d = d[d['horsepower']!='?']

# Constructing the X and Y matrices
features = ['horsepower', 'weight', 'acceleration']

for ft in features:
    d[ft] = pd.to_numeric(d[ft])

X = d[features]
Y = d['mpg'].values.tolist()

# Initiating the Node
root = NodeRegression(Y, X, max_depth=3, min_samples_split=3)

# Growing the tree
root.grow_tree()

# Printing tree 
root.print_tree()
```