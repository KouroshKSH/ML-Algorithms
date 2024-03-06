import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node():
    def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):
        # decision nodes
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right

        # leaf nodes
        self.value = value


class DecisionTree():
    def __init__(self, min_sample_split=3, max_depth=3):
        self.root = None

        # cut-off for splitting
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def build_tree(self, data, current_depth=0):
        # x should not have the diagnosis column while y should be the diagnosis column
        x, y = data[:, 1:], data[:, 0]
        number_samples, number_features = np.shape(x)

        # split until stopping conditions
        if number_samples >= self.min_sample_split and current_depth <= self.max_depth:
            # find the best attribute for splitting
            best_attribute = self.get_best_attribute(data,
                                                     number_samples,
                                                     number_features)

            if best_attribute["info_gain"] > 0:
                # make left subtree
                left_tree = self.build_tree(best_attribute["left_data"],
                                            current_depth + 1)

                # make right subtree
                right_tree = self.build_tree(best_attribute["right_data"],
                                             current_depth + 1)

                # decision node
                return Node(best_attribute["feature_idx"],
                            best_attribute["threshold"],
                            left_tree, right_tree,
                            best_attribute["info_gain"])

        # leaf nodes
        leaf_value = self.get_leaf_value(y)
        return Node(value=leaf_value)


    def get_best_attribute(self, data, samples, features):
        max_info_gain = -float("inf") # start with the minimum possible number
        best_attribute = {}

        # check all features to find the best one for splitting
        for feature_idx in range(features):
            feature_values = data[:, feature_idx]

            # we only want present values, not all the real numbers (infinitly many)
            candidate_thresholds = np.unique(feature_values)

            # check feature values
            for threshold in candidate_thresholds:
                # make left and right subtrees
                left_data, right_data = self.split(data, feature_idx, threshold)

                # we want non-empty children/subtrees
                if len(left_data) > 0 and len(right_data) > 0:
                    # the diagnosis column is the first column, hence `[:, 0]`
                    y, left_y, right_y = data[:, 0], left_data[:, 0], right_data[:, 0]

                    # use gini instead of entropy since it has fewer computations
                    current_info_gain = self.information_gain(y, left_y, right_y, "gini")

                    # when we find a better attribute using info gain
                    if current_info_gain > max_info_gain:
                        # update the values for the keys of the dictionary
                        best_attribute["feature_idx"] = feature_idx
                        best_attribute["threshold"] = threshold
                        best_attribute["left_data"] = left_data
                        best_attribute["right_data"] = right_data
                        best_attribute["info_gain"] = current_info_gain

                        # update max info gain for next round
                        max_info_gain = current_info_gain

        return best_attribute


    def split(self, data, feature_idx, threshold):
        left_data = np.array([i for i in data if i[feature_idx] < threshold])
        right_data = np.array([i for i in data if i[feature_idx] >= threshold])
        return left_data, right_data


    def entropy_measure(self, y):
        entropy = 0
        labels = np.unique(y)
        for label in labels:
            # calculate the conditional probability (i.e., "y given label")
            p = len(y[y == label]) / len(y)
            entropy += p * np.log2(1 / round(p, 3)) # to avoid extra computation

        return entropy

    def gini_measure(self, y):
        gini = 0
        labels = np.unique(y)
        for label in labels:
            # calculate the conditional probability (i.e., "y given label")
            p = len(y[y == label]) / len(y)
            gini += np.power(p, 2)

        return 1 - gini


    def information_gain(self, parent, left_child, right_child, measure="gini"):
        # compute IG with respect to the weights of the subtrees
        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)

        if measure == "gini":
            gain = self.gini_measure(parent) - \
                    (left_weight * self.gini_measure(left_child) + \
                    right_weight * self.gini_measure(right_child))
        else:
            gain = self.entropy_measure(parent) - \
                    (left_weight * self.entropy_measure(left_child) + \
                    right_weight * self.entropy_measure(right_child))
        return gain

    def get_leaf_value(self, y):
        Y = list(y) # to make it mutable
        return max(Y, key=Y.count) # find the most frequent class label

    def fit(self, x, y):
        data = np.concatenate((x, y), axis=1)
        self.root = self.build_tree(data)

    def predict(self, X):
        p = [self.make_prediction(x, self.root) for x in X]
        return p

    def make_prediction(self, x, decision_tree):
        if decision_tree.value != None:
            return decision_tree.value

        feature_value = x[decision_tree.feature_idx]
        if feature_value < decision_tree.threshold:
            return self.make_prediction(x, decision_tree.left)
        else:
            return self.make_prediction(x, decision_tree.right)


    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("x_"+str(tree.feature_idx), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


data = pd.read_csv("breast_cancer_dataset.csv")
print(data.info())

# let's drop the columns we don't need/want
data.drop(columns=["id", "Unnamed: 32"], inplace=True)
data = data[:100]
print(data.describe())

x = data.iloc[:, 1:].values # x should not have access to diagnosis column
y = data.iloc[:, 0].values.reshape(-1, 1) # y should have the reshaped diagnosis

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2, random_state=42)

classifier = DecisionTree(min_sample_split=3, max_depth=3)
classifier.fit(x_train, y_train)

classifier.print_tree()

y_pred = classifier.predict(x_test)
accuracy_score(y_test, y_pred)
