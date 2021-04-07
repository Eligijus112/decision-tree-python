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
        level: int, 
        Y: list 
    ):
        self.level = level
        self.Y = Y 

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)

        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Sorting the counts and saving the final prediction of the node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        yhat = counts_sorted[-1][0]

        # Saving to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat 

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        # Getting the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0), self.counts.get(1)

        # Getting the GINI impurity
        return GINI_impurity(y1_count, y2_count)


class DecisionTree:
    
    def __init__(
        self, 
        max_depth: int,
        min_node_obs: int 
        )
        # Defining the maximum depth a tree can grow
        self.max_depth = max_depth

        # Definining the min number of observations in each node for a split to commence 
        self.min_node_obs = min_node_obs
        
        # When initializing the object, set the depth to 0 
        self.depth = 0 

    @staticmethod
    def GINI_impurity(y1_count, y2_count):
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        # Getting the total observations
        n = y1_count + y2_count
        
        # Getting the probability to see each of the classes
        p1 = y1_count / n
        p2 = y2_count / n
        
        # Calculating GINI 
        gini = 1 - (p1 ** 2 + p2 ** 2)
        
        # Returning the gini impurity
        return gini

    def get_nodes(
        self, 
        X: pd.DataFrame,
        Y: list
        ):
        """
        Function that returns left and right nodes (two X and Y tuples)
        """
        # Saving the features
        features = list(X.columns)

        # Iterating through the features 

    @staticmethod
    def GINI_impurity_df_split(
        df:pd.DataFrame, 
        target, 
        split_X, 
        split_value
    ):
        """
        Function to calculate the resulting GINI impurity of a split
        """
        # Getting the left and right nodes
        left = df[df[split_X] <= split_value]
        right = df[df[split_X] > split_value]
        
        if (min_n > left.shape[0]):
            return 0.5
        
        # Getting the counts and ginis
        left_counts = left.groupby(target).size().values.tolist()
        right_counts = right.groupby(target).size().values.tolist()

        gini_left = GINI_impurity(left_counts[0], left_counts[1])
        gini_right = GINI_impurity(right_counts[0], right_counts[1])
        
        # Getting the final weighted GINI impurity
        w1 = left.shape[0]/ (left.shape[0] + right.shape[0]) 
        w2 = right.shape[0] / (left.shape[0] + right.shape[0]) 
        
        return w1 * gini_left + w2 * gini_right


if __name__ == '__main__':
    # Reading data
    d = pd.read_csv("data/train.csv")

    # Constructing the X and Y matrices
    X = d[['Age', 'Fare']]
    Y = d['Survived'].values.tolist()

    # Creating the node object 
    node = Node(level=0, Y=Y)