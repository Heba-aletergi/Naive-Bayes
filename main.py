import numpy as np
import pandas as pd
import array
import NaiveBayes_1 as NaiveBayes


def prepare_data(data):
   '''
        Split data into X and Y
   '''
   X = data.drop(data.columns[1], axis=1)
   Y = data[data.columns[1]]
   return X , Y

def main():
    #y_train = np.array(['A','B','B','A','B','A','A','B','B'])
    train_data = pd.read_csv('breast-cancer-training.csv')
    x_train, y_train = prepare_data(train_data)
    #x_train = data.drop(data.columns[1], axis=1)
    #y_train = data[data.columns[1]]
    
    clf = NaiveBayes.NaiveBayesClassifier()
    clf.train(x_train.iloc[:, 1:], y_train.values)

    # Pre-pared test data 
    test_data = pd.read_csv('breast-cancer-test.csv')
    x_test, y_test = prepare_data(test_data)
    #x_test = data.drop(data.columns[1], axis=1)
    #y_test = data[data.columns[1]]

    # Make prediction on test data 
    predictions, outcome_probs = clf.predict(x_test.iloc[:, 1:])
    # [idx][0] score of (no-recurrence-events) , [idx][1] score of (recurrence-events)
    print()

if __name__ == '__main__':
  main()

