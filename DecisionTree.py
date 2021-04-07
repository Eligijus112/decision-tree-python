# Data wrangling 
import pandas as pd 

# Array math
import numpy as np 

# Quick value count calculator
from collections import Counter


def GINI_impurity(y1_count: int, y2_count: int) -> float:
    """
    Given the observations of a binary class calculate the GINI impurity
    """
    # Getting the total observations
    n = y1_count + y2_count
    
    # If n is 0 then we return the lowest possible gini impurity
    if n == 0:
        return 0.0

    # Getting the probability to see each of the classes
    p1 = y1_count / n
    p2 = y2_count / n
    
    # Calculating GINI 
    gini = 1 - (p1 ** 2 + p2 ** 2)
    
    # Returning the gini impurity
    return gini


class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(
        self, 
        Y: list 
    ):
        self.Y = Y 

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)

        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat 

        # Saving the number of observations in the node 
        self.n = len(Y)

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        # Getting the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        # Getting the GINI impurity
        return GINI_impurity(y1_count, y2_count)



class DecisionTree:
    
    def __init__(
        self, 
        max_depth: int,
        min_node_obs: int 
        ):
        # Defining the maximum depth a tree can grow
        self.max_depth = max_depth

        # Definining the min number of observations in each node for a split to commence 
        self.min_node_obs = min_node_obs
        
        # When initializing the object, set the depth to 0 
        self.depth = 0 

    def best_split(self, X: pd.DataFrame, Y: list) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = X.copy()
        df['Y'] = Y

        # Extracting all the features
        features = list(X.columns)

        # Getting the GINI impurity for the base input 
        counts = Counter(Y)
        GINI_base = GINI_impurity(counts.get(0), counts.get(1))

        # Finding which split yields the best GINI gain 
        max_gain = 0

        # Default best feature and split
        best_feature = features[0]
        best_value = 0

        for feature in features:
            # Droping missing values
            Xvalues = X[feature].dropna()

            # Sorting the values and getting the rolling average
            Xvalues = Xvalues.sort_values().rolling(2).mean()

            # Converting to list
            Xvalues = Xvalues.values.tolist()
            
            # Droping the initial NaN value 
            Xvalues.pop(0)

            for value in Xvalues:
                # Spliting the dataset 
                left_df = df[df[feature]<value]
                right_df = df[df[feature]>=value]

                # Creating the two nodes 
                left_node = Node(left_df['Y'].values.tolist())
                right_node = Node(right_df['Y'].values.tolist())

                # Calculating the weights for each of the nodes
                w_left = left_node.n / (left_node.n + right_node.n)
                w_right = right_node.n / (left_node.n + right_node.n)

                # Calculating the weighted GINI impurity
                wGINI = w_left * left_node.gini_impurity + w_right * right_node.gini_impurity

                # Calculating the GINI gain 
                GINIgain = GINI_base - wGINI

                # Checking if this is the best split so far 
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    max_gain = GINIgain

        return (best_feature, best_value)



if __name__ == '__main__':
    # Reading data
    d = pd.read_csv("data/train.csv")

    # Constructing the X and Y matrices
    X = d[['Age', 'Fare']]
    Y = d['Survived'].values.tolist()

    # Creating the node object 
    DT = DecisionTree(max_depth=2, min_node_obs=5)

    # Gettting the best split
    (feature, split) = DT.best_split(X, Y)

    print((feature, split))