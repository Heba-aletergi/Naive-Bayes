import numpy as np
import pandas as pd
import array
from collections import Counter


class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = None
        self.feature_probabilities = None

    def train(self, X_train, y_train):
        num_instances, num_features = X_train.shape
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)

        # Calculate class probabilities
        self.class_probabilities = self.calculate_class_probabilities(y_train, unique_classes)

        # Calculate feature probabilities for each class
        self.feature_probabilities = []
        for class_label in unique_classes:  # Iterate over each unique class
            subset_X = X_train[y_train == class_label]
            feature_probabilities = []
            for feature_index in range(num_features):  # Iterate over each feature
                feature_values = np.unique(X_train.iloc[:, feature_index])
                feature_counts = self.calculate_feature_counts(subset_X.iloc[:, feature_index], feature_values)
                feature_probabilities.append(feature_counts / len(subset_X))
            self.feature_probabilities.append(feature_probabilities)

    def calculate_class_probabilities(self, y_train, unique_classes):
        class_probabilities = {}
        for class_label in unique_classes:
            class_count = np.sum(y_train == class_label)
            class_probabilities[class_label] = class_count / len(y_train)
        return class_probabilities

    def calculate_feature_counts(self, feature_values, unique_values):
        feature_counts = []
        for value in unique_values:
            feature_counts.append(np.sum(feature_values == value))
        return feature_counts

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