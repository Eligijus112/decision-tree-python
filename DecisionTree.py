# Data wrangling 
import pandas as pd 

# Array math
import numpy as np 

# Quick value count calculator
from collections import Counter


class Node: 
    """
    Class for creating the nodes for a decision tree 
    """
    def __init__(
        self, 
        Y: list,
        X: pd.DataFrame,
        min_obs_child=None,
        max_depth=None,
        depth=None
    ):
        # Saving the data to the node 
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_obs_child = min_obs_child if min_obs_child else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of node 
        self.depth = depth if depth else 0

        # Extracting all the features
        self.features = list(self.X.columns)

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

        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        # Ensuring the correct types
        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

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

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node 
        """
        # Getting the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        # Getting the GINI impurity
        return self.GINI_impurity(y1_count, y2_count)

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split 
        for a decision tree
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the GINI impurity for the base input 
        GINI_base = self.get_GINI()

        # Finding which split yields the best GINI gain 
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xvalues = self.X[feature].dropna()

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
                left_node = Node(left_df['Y'].values.tolist(), left_df[self.features])
                right_node = Node(right_df['Y'].values.tolist(), right_df[self.features])

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

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data 
        df = self.X.copy()
        df['Y'] = self.Y

        # If there is GINI to be gained, we split further 
        if (self.n >= self.min_obs_child) and (self.depth < self.max_depth):

            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<best_value].copy(), df[df[best_feature]>=best_value].copy()

                # Creating the left and right nodes
                left = Node(left_df['Y'].values.tolist(), left_df[self.features], depth=self.depth + 1, max_depth=self.max_depth, min_obs_child=self.min_obs_child)
                right = Node(right_df['Y'].values.tolist(), right_df[self.features], depth=self.depth + 1, max_depth=self.max_depth, min_obs_child=self.min_obs_child)

                # Saving the left and right nodes to the current node 
                self.left = left 
                self.right = right

                # Spliting the left and right nodes further 
                self.left.grow_tree()
                self.right.grow_tree()

    def print_info(self):
        """
        Method to print the infromation about the tree
        """
        print(f"-------")
        print(f"Depth of the node: {self.depth}")
        print(f"GINI impurity of the node: {self.gini_impurity}")
        print(f"Class distribution in the node: {self.counts}")
        print(f"Feature to split on: {self.best_feature}")
        print(f"Feature value to split on: {self.best_value}")
        print(f"-------")

    def print_tree(self):
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()


if __name__ == '__main__':
    # Reading data
    d = pd.read_csv("data/train.csv")[['Age', 'Fare', 'Survived']].dropna()

    # Constructing the X and Y matrices
    X = d[['Age', 'Fare']]
    Y = d['Survived'].values.tolist()

    # Initiating the Node
    root = Node(Y, X, max_depth=3)

    # Getting teh best split
    root.grow_tree()

    # Printing the tree information 
    root.print_tree()

    # Creating the node object 
    #DT = DecisionTree(max_depth=2, min_node_obs=5)

    # Gettting the best split
    #(feature, split) = DT.best_split(X, Y)