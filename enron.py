'''
 UCI CS177: Spam email prediction
 xxxxxxxxxx
 This is a template file for code submission:
 - It begins with an example of how to load the Enron email 
   training & test data, and how to compute simple statistics of this data.
 - Your task is to fill in the functions compute_word_probs, 
   word_classifier, and compute_accuracy
 - Do not change the name of the file: it should be submitted as enron.py
 - Upon submitting your code to Gradescope, an autograder will run and inform
   you if there are any errors compiling your code. It can take a minute, so
   please be patient so you can confirm your code is formatted correctly.
 - Each student should write code independently. Refer to the Canvas page on 
   Collaboration and Academic Honesty for more details.
'''

import numpy as np

# Load data
def load_data():
    with np.load('enron.npz') as data:
        return dict(**data)
    
data = load_data()
trainFeat = data['trainFeat']
trainLabels = data['trainLabels']
testFeat = data['testFeat']
testLabels = data['testLabels']
vocab = data['vocab']
W = len(vocab)
'''
    Data description:
    - trainFeat: (Dtrain, W) logical 2d-array of word appearance for training documents.
    - trainLabels: (Dtrain,) 1d-array of {0,1} training labels where 0=ham, 1=spam.
    - testFeat: (Dtest, W) logical 2d-array of word appearance for test documents.
    - testLabels:  (Dtest,) 1d-array of {0,1} test labels where 0=ham, 1=spam.
    - vocab: (W,) 1d-array where vocab[i] is the English characters for word i.
'''

# Separate "ham" and "spam" classes, subsample selected vocabulary words
vocabInds = np.arange(W) 
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=0) can be computed from countsHam and numHam
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=1) can be computed from countsSpam and numSpam

# Display words that are common in one class, but rare in the other
ind = np.argsort(countsHam-countsSpam)
print('Words common in Ham but not Spam:')
for i in range(-1, -100, -1):
    print(vocab[ind[i]], end=' ')
print()
print('Words common in Spam but not Ham:')
for i in range(100):
    print(vocab[ind[i]], end=' ')









# Begin Assigment:

def compute_word_probs(features, labels):
    '''
    Computes P(X_i | Y) for each word i in the vocabulary.
    
    Parameters
    ----------
    features :
       Matrix for the training set where each row indicates the 
       words present in an email.
       
    labels :
        Vector of the labels for each email in the training set.
        
    Returns
    -------
    probs :
        A 2 x |vocab| matrix, where the first row contains P(X_i | Y = ham)
        for each vocab word and the second row contains P(X_i | Y = spam)
        
    '''
    probs = None
    return probs


def word_classifier(probs, test, vocab=[], words=[]):
    '''
    Predicts whether each email in the test set is spam.
    
    Parameters
    ----------
    probs :
       The matrix of conditional probabilities estimated from the training
       set using compute_word_probs.
       
    test :
        The feature matrix for the test set.

    vocab :
        The list of vocab words. Only needs to be specified when words is non-empty.

    words :
        The subset of words to use for classification. (If empty, use all)
        
    Returns
    -------
    predictions :
        A vector of predicted labels for the test set
        
    '''
    predictions = None
    return predictions

def compute_accuracy(predictedLabels, trueLabels):
    '''
    Computes the accuracy of our test set predictions.
    
    Parameters
    ----------
    predictedLabels :
        The labels predicted by our classifier.
       
    trueLabels :
        The ground-truth labels included in the test set.
        
    Returns
    -------
    accuracy :
        The fraction of correctly classified emails.
    '''
    accuracy = None
    return accuracy
    

# 4b: Compute the probabilities used by our classifier
probs = compute_word_probs(trainFeat, trainLabels)

# Run the classifier using the words specified for each question 
predictedMoney = word_classifier(probs, testFeat, vocab, ['money'])
print('4c) Accuracy using word "money" is ', \
     compute_accuracy(predictedMoney, testLabels))

predictedThanks = word_classifier(probs, testFeat, vocab, ['thanks'])
print('4d) Accuracy using word "thanks" is ', \
     compute_accuracy(predictedThanks, testLabels))

predictedPossibilities = word_classifier(probs, testFeat, vocab, ['possibilities'])
print('4e) Accuracy using word "possibilities" is ', \
     compute_accuracy(predictedPossibilities, testLabels))

predicted3Words = word_classifier(probs, testFeat, vocab, ['money', 'thanks', 'possibilities'])
print('4f) Accuracy using 3 words is ', \
     compute_accuracy(predicted3Words, testLabels))
        
predictedAll = word_classifier(probs, testFeat)
print('4g) Accuracy using all words is ', \
     compute_accuracy(predictedAll, testLabels))
