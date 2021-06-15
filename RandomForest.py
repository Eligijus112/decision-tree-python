"""
Code that houses the class that creates and uses the random forest classifier 
"""
# Data wrangling 
import pandas as pd 

# Numerical operations
import numpy as np 

# Random selections
import random 

# Quick value count calculator
from collections import Counter

# Tree growth tracking 
from tqdm import tqdm

# Accuracy metrics 
from sklearn.metrics import precision_score, recall_score


class RandomForestTree():
    """
    Class that grows one random forest tree
    """
    def __init__(
        self,
        Y, 
        X, 
        min_samples_split=None,
        max_depth=None,
        depth=None,
        X_features_fraction=None,
        node_type=None,
        rule=None
    ):
        # Saving the data for the random forest
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of the tree
        self.depth = depth if depth else 0

        # Saving the final feature list 
        self.features = list(X.columns)

         # Type of node 
        self.node_type = node_type if node_type else 'root'

        # Rule for spliting 
        self.rule = rule if rule else ""

        # Calculating the counts of Y in the node 
        self.counts = Counter(Y)

        # Getting the GINI impurity based on the Y distribution
        self.gini_impurity = self.get_GINI()

        # Getting the number of features 
        self.n_features = len(self.features)

        # Saving the hyper parameters specific to the random forest 
        self.X_features_fraction = X_features_fraction if X_features_fraction is not None else 1.0

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
    
    def get_random_X_colsample(self):
        # Getting the random subset of features 
        n_ft = int(self.n_features * self.X_features_fraction)

        # Selecting random features without repetition
        features = random.sample(self.features, n_ft)

        # Subseting the X to chosen features 
        X = self.X[features].copy()

        # Returning the subseted features
        return X 

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

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window

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

        # Getting a random subsample of features 
        n_ft = int(self.n_features * self.X_features_fraction)

        # Selecting random features without repetition
        features_subsample = random.sample(self.features, n_ft)

        for feature in features_subsample:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Spliting the dataset 
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])

                # Getting the Y distribution from the dicts
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                # Getting the left and right gini impurities
                gini_left = self.GINI_impurity(y0_left, y1_left)
                gini_right = self.GINI_impurity(y0_right, y1_right)

                # Getting the obs count from the left and the right data splits
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right

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
        # If there is GINI to be gained, we split further 
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                # Getting the left and right dataframe indexes
                left_index, right_index = self.X[self.X[best_feature]<=best_value].index, self.X[self.X[best_feature]>best_value].index

                # Extracting the left X and right X 
                left_X, right_X = self.X[self.X.index.isin(left_index)], self.X[self.X.index.isin(right_index)]

                # Reseting the indexes
                left_X.reset_index(inplace=True, drop=True)
                right_X.reset_index(inplace=True, drop=True)

                # Extracting the left Y and the right Y 
                left_Y, right_Y = [self.Y[x] for x in left_index], [self.Y[x] for x in right_index]

                # Creating the left and right nodes
                left = RandomForestTree(
                    left_Y, 
                    left_X, 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split, 
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                self.left = left 
                self.left.grow_tree()

                right = RandomForestTree(
                    right_Y, 
                    right_X, 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                self.right = right
                self.right.grow_tree()
    
    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if (cur_node.n < cur_node.min_samples_split) | (best_feature is None):
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return cur_node.yhat

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | GINI impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()


class RandomForestClassifier():
    """
    Class that creates a random forest for classification problems
    """
    def __init__(
        self,
        Y: list,
        X: pd.DataFrame,
        min_samples_split=None,
        max_depth=None,
        n_trees=None,
        X_features_fraction=None,
        X_obs_fraction=None
    ):  
        # Saving the data for the random forest
        self.Y = Y 
        self.X = X

        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Saving the final feature list 
        self.features = list(X.columns)

        # Getting the number of features 
        self.n_features = len(self.features)

        # Saving the hyper parameters specific to the random forest 
        self.n_trees = n_trees if n_trees is not None else 30
        self.X_features_fraction = X_features_fraction if X_features_fraction is not None else 1.0
        self.X_obs_fraction = X_obs_fraction if X_obs_fraction is not None else 1.0

    def bootstrap_sample(self):
        """
        Function that creates a bootstraped sample with the class instance parameters 
        """
        # Sampling the number of rows with repetition
        Xbootstrap = self.X.sample(frac=self.X_obs_fraction, replace=True) 

        # Getting the index of samples 
        indexes = Xbootstrap.index

        # Getting the corresponding Y variables
        Ybootstrap = [self.Y[x] for x in indexes]

        # Droping the index of X 
        Xbootstrap.reset_index(inplace=True, drop=True)

        # Returning the X, Y pair
        return Xbootstrap, Ybootstrap

    def grow_random_forest(self):
        """
        Main method of the class; Creates **n_trees** random trees
        """
        # List to hold trees in 
        random_forest = []

        # Iterating 
        for _ in tqdm(range(self.n_trees)):
            # Getting the bootstrapped sample
            X, Y = self.bootstrap_sample()
            
            # Initiating the random tree
            tree = RandomForestTree(
                Y=Y, 
                X=X, 
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                X_features_fraction=self.X_features_fraction
                )

            # Growing the tree
            tree.grow_tree()

            # Appending the tree to the list of trees (the forest)
            random_forest.append(tree)
        
        # Saving the random forest list to memory
        self.random_forest = random_forest

    def print_trees(self):
        """
        Method to print out all the grown trees in the classifier 
        """
        for i in range(self.n_trees):
            print("------ \n")
            print(f"Tree number: {i + 1} \n")
            self.random_forest[i].print_tree()
            print("------ \n")

    def tree_predictions(self, X: pd.DataFrame) -> list:
        """
        Method to get the predictions from all the trees 
        """
        predictions = []
        for i in range(self.n_trees):
            yhat = self.random_forest[i].predict(X)
            
            # Apending to prediction placeholder
            predictions.append(yhat)
        
        # Returning the prediction list 
        return predictions
    
    def predict(self, X: pd.DataFrame) -> list:
        """
        Method to get the final prediction of the whole random forest 
        """
        # Getting the individual tree predictions
        yhat = self.tree_predictions(X)

        # Saving the number of obs in X 
        n = X.shape[0]

        # Getting the majority vote of each coordinate of the prediction list 
        yhat_final = []

        for i in range(n):
            yhat_obs = [x[i] for x in yhat]

            # Getting the most frequent entry 
            counts = Counter(yhat_obs)
            most_common = counts.most_common(1)[0][0]

            # Appending the most common entry to final yhat list 
            yhat_final.append(most_common)
        
        # Returning the final predictions 
        return yhat_final

if __name__ == '__main__':
    # Reading data for classification 
    d = pd.read_csv("data/random_forest/telecom_churn.csv")

    # Setting the features used 
    features = [
        'AccountWeeks',
        'DataUsage',
        'DayMins',
        'DayCalls',
        'MonthlyCharge',
        'OverageFee',
        'RoamMins'
    ]

    # Initiating the random forest object 
    rf = RandomForestClassifier(
        Y=d['Churn'], 
        X=d[features],
        min_samples_split=5,
        max_depth=4,
        n_trees=10,
        X_features_fraction=0.5
        )
    
    # Growing the random forest 
    rf.grow_random_forest()

    # Printing out the trees 
    rf.print_trees()

    # Making predictions
    yhat = rf.predict(d[features])
    d['yhat'] = yhat 

    # Measurring accuracy
    print(f"The training precision: {precision_score(d['Churn'], d['yhat'])}")
    print(f"The training recall: {recall_score(d['Churn'], d['yhat'])}")