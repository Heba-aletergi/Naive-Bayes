import numpy as np
import pandas as pd
import array
import NaiveBayes_1 as NaiveBayes

def export_file(predicts, score):
   '''
        Export prediction result into (Sasmpleoutput.txt)
   '''
   with open("sampleoutput.txt", "w") as file:
      for predict_class, score_, index in zip(predicts, score, range(len(predicts))):
        class_score = score_[0] if predict_class == 'no-recurrence-events' else score_[1]
        contect = "Instance ("+ str(index) +"):\n Predicted class is ("+ str(predict_class) +"), its score is ("+ str(class_score) +") \n\n" 
        file.writelines(contect)

def print_prediction(predicts, score):
   '''
        Print prediction result 
   '''
   for predict_class, score_, index in zip(predicts, score, range(len(predicts))):
      print(f'Instance({index}):\n Predicted class is ({predict_class})\n Class score: (no-recurrence-events: {"{:.5f}".format(score_[0])}, recurrence-events:{"{:.5f}".format(score_[1])})))')

def print_probability(features_prob, class_probs, feature_names):
    ''' 
        Print Conditional probability for each feature
    '''
    # Print feature Probability
    for feat in feature_names:  # Iterate over each feature
        print(f'CONDITIONAL PROBANILITY OF FEATURE ({feat}):\n')
        for class_label in features_prob.keys(): # Iterate over each class label
        #print(f'Conditional probability for class ({class_label}):\n')

            for feat_val in features_prob[class_label][feat].keys():
                print(f' P( {feat}=({feat_val}) | Y=({class_label}) ) = {"{:.5f}".format(features_prob[class_label][feat][feat_val])} \n')
        print('------------------------------------------------')
    print('------------------------------------------------------------------------------------------')
    # Print class probability
    print(f'THE PROBABILITY OF CLASS LABEL')
    for class_label in class_probs:
       print(f'P ({class_label}): {"{:.5f}".format(class_probs[class_label])}\n')

def prepare_data(data):
   '''
        Split data into X and Y
   '''
   X = data.drop(data.columns[1], axis=1)
   Y = data[data.columns[1]]
   return X , Y

def main():
    # Pre-pared train data 
    train_data = pd.read_csv('breast-cancer-training.csv')
    x_train, y_train = prepare_data(train_data)

    # Naive Bayes classifier
    clf = NaiveBayes.NaiveBayesClassifier()
    clf.train(x_train.iloc[:, 1:], y_train.values)
    
    # Pre-pared test data 
    test_data = pd.read_csv('breast-cancer-test.csv')
    x_test, y_test = prepare_data(test_data)

    # Make prediction on test data 
    predictions, outcome_probs = clf.predict(x_test.iloc[:, 1:])
    # HINT: for(outcome_probs): [idx][0] score of (no-recurrence-events) , [idx][1] score of (recurrence-events)

    # Export score and class predicted into file(sampleoutput.txt)
    export_file(predictions, outcome_probs)

    # Print out: conditional prob for each feature:(Condirtion Prob, Possible values) & Prob of each class label
    print_probability(clf.feature_probabilities, clf.class_probabilities, x_train.columns[1:])
    print('------------------------------------------------------------------------------------------')
    # print score of (no-recurrence-events, recurrence-events) and prediction class for each test instance.
    print_prediction(predictions, outcome_probs)


if __name__ == '__main__':
  main()

