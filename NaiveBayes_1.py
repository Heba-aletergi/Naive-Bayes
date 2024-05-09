import numpy as np
import pandas as pd
import array
from collections import Counter, defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = None
        self.feature_probabilities = None
        self.feature_priors = None

    def train(self, X_train, y_train):
        num_instances, num_features = X_train.shape
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)

        # Calculate class probabilities
        self.class_probabilities = self.calculate_class_probabilities(y_train, unique_classes)

        # Calculate feature probabilities for each class
        self.feature_probabilities = defaultdict(dict)
        for class_label in unique_classes:  # Iterate over each unique class
            subset_X = X_train[y_train == class_label]
            feature_probabilities = defaultdict(dict)
            for feature_index in range(num_features):  # Iterate over each feature
                feature_values = np.unique(X_train.iloc[:, feature_index])  # get unique val for feature (feature_index)
                feature_values_counts = self.calculate_feature_counts(subset_X.iloc[:, feature_index], feature_values)
                feature_probabilities[X_train.columns[feature_index]] = feature_values_counts
            self.feature_probabilities[class_label] = feature_probabilities

        # Claulate prior for each feature that correspond to class (class_label)
        self.feature_priors = defaultdict(dict)
        self.feature_priors = self.calculate_prior_features(X_train, num_features)

    def calculate_class_probabilities(self, y_train, unique_classes):
        class_probabilities = {}
        for class_label in unique_classes:
            class_count = np.sum(y_train == class_label)
            class_probabilities[class_label] = class_count / len(y_train)
        return class_probabilities

    def calculate_feature_counts(self, feature_values, unique_values):
        feature_counts = {}
        for value in unique_values:  # Iterate over unique val of feature
            value_count = np.sum(feature_values == value)
            feature_counts[value] = value_count /len(feature_values)
        return feature_counts

    def calculate_prior_features(self, x_train, num_features):
        prior_features = defaultdict(dict)
        data_size = len(x_train)
        for feature_index in range(num_features):
            feat_vals = x_train.iloc[:,feature_index].value_counts().to_dict()  # count uniqe each value in feature (feature)
            for feat_val, count in feat_vals.items():   # Iterate over each unique value
                prior_features[x_train.columns[feature_index]][feat_val] = count / data_size
        return prior_features

    def predict(self, X_test):
        predictions = []
        for instance in X_test:
            likelihoods = []
            for class_label, class_prob in enumerate(self.class_probabilities):
                likelihood = class_prob
                for feature_index, feature_value in enumerate(instance):
                    feature_probs = self.feature_probabilities[class_label][feature_index]
                    likelihood *= feature_probs[feature_value]
                likelihoods.append(likelihood)
            predicted_class = np.argmax(likelihoods)
            predictions.append(predicted_class)
        return predictions

def test():
    #y_train = np.array(['A','B','B','A','B','A','A','B','B'])
    data = pd.read_csv('breast-cancer-training.csv')
    x_train = data.drop(data.columns[1], axis=1)
    y_train = data[data.columns[1]]
    
    clf = NaiveBayesClassifier()
    clf.train(x_train.iloc[:, 1:], y_train.values)

if __name__ == '__main__':
  test()