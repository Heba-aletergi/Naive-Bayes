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
        self.class_probabilities = self.calculate_class_prob(y_train, unique_classes)

        # Calculate feature probabilities for each class
        self.feature_probabilities = defaultdict(dict)
        for class_label in unique_classes:  # Iterate over each unique class
            subset_X = X_train[y_train == class_label]
            feature_probabilities = defaultdict(dict)
            for feature_index in range(num_features):  # Iterate over each feature
                feature_values = np.unique(X_train.iloc[:, feature_index])  # get unique val for feature (feature_index)
                feature_values_counts = self.calculate_features_likehood(subset_X.iloc[:, feature_index], feature_values)
                feature_probabilities[X_train.columns[feature_index]] = feature_values_counts
            self.feature_probabilities[class_label] = feature_probabilities

        # Claulate prior for each feature that correspond to class (class_label)
        self.feature_priors = defaultdict(dict)
        self.feature_priors = self.calculate_features_prior(X_train, num_features)

    def calculate_class_prob(self, y_train, unique_classes):
        class_probabilities = {}
        for class_label in unique_classes:
            class_count = np.sum(y_train == class_label)
            class_probabilities[class_label] = class_count / len(y_train)
        return class_probabilities

    def calculate_features_likehood(self, feature_values, unique_values):
        feature_counts = {}
        for value in unique_values:  # Iterate over unique val of feature
            value_count = np.sum(feature_values == value)
            feature_counts[value] = value_count /len(feature_values)
        return feature_counts

    def calculate_features_prior(self, x_train, num_features):
        prior_features = defaultdict(dict)
        data_size = len(x_train)
        for feature_index in range(num_features):
            feat_vals = x_train.iloc[:,feature_index].value_counts().to_dict()  # count uniqe each value in feature (feature)
            for feat_val, count in feat_vals.items():   # Iterate over each unique value
                prior_features[x_train.columns[feature_index]][feat_val] = count / data_size
        return prior_features

    def predict(self, X_test):
        predictions = []
        outcome_probs = []
        x = np.array(X_test)

        for instance in x:
            probs_outcome = {}

            likelihood = 1
            feature_prior = 1
            #for class_prob, class_label in enumerate(self.class_probabilities):
            for class_label, class_prob in self.class_probabilities.items():  # Iterate over Class ()
                class_prior = class_prob
                for feature_index, feature_value in enumerate(instance):
                    likelihood *= self.feature_probabilities[class_label][X_test.columns[feature_index]][feature_value]
                    feature_prior *= self.feature_priors[X_test.columns[feature_index]][feature_value]

                posterior = (likelihood * class_prior) / (feature_prior)  # Calcualte postirior
                probs_outcome[class_label] = posterior

            predict_class = max(probs_outcome, key = lambda x: probs_outcome[x]) # compare and get class of hieghts value
            outcome_probs.append(list(probs_outcome.values()))  
            predictions.append(predict_class)

            #predicted_class = np.argmax(likelihoods)
            #predictions.append(predicted_class)

        return predictions, outcome_probs