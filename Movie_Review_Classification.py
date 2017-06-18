import nltk.classify.util
import random
import pickle
from nltk.classify import apply_features
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)


def saveClassifier(classifier, fileName):
    '''
    Given a classifier and filename, serializes classifier object
    
    Arguments:
        classifier:          Classifier to be saved
        fileName:            Name of file to be saved
    
    '''
    save_classifier = open(fileName, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
    return 

def loadClassifier(fileName):
    '''
    Loads and returns a classifier given the filename
    
    Arguments:
        fileName:         Name of file to load 
    '''
    
    classifier_f = open(fileName, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

def loadReviews():    
    '''
    Loads all reviews. Final list contains tuples of format:
    (list of words contained in file, neg/pos).
    '''
    allReviews = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
    return allReviews

def loadTrainingTesting(allData, percentageTest):
    '''
    Given list of tuples consisting of (features, label), shuffles the data
    and allocates certain amount of data to training and certain amount of data 
    to test depending on what user wants 
    
    
    Arguments:
        allData:          List of tuples of form (feature, classification)
        percentageTest:   Proportion of data allocated for testing
        
    '''
    random.shuffle(allData)
    threshold = int(percentageTest * len(allData))
    testingData = allData[:threshold]
    trainingData = allData[threshold:]
    return trainingData, testingData

def returnAllWords():
    '''
    Returns list of all words seen in movie reviews. This list is expected
    to have repeated words. 
    '''
    
    allWords = []
    for w in movie_reviews.words():
        allWords.append(w.lower())
    return allWords

def returnFreqMapOfWords(words):
    '''
    Given a list of words, return a dictionary that maps each word to its 
    frequency

    Arguments:
        words:          List of words to make frequency dictionary of 
    '''    
    freqMap = nltk.FreqDist(words)
    return freqMap

def returnTopWords(freqMap, n):
    '''
    Given a mapping of word to frequency, returns list of n random words
    
    Arguments:
        freqMap:          Frequency map of word to number of occurences
        n:                Number of random words to be returned 
    '''
    
    return list(freqMap.keys())[:n]
    
    
def returnFeatureForDocument(document, bagOfWords):
    '''
    Given all words of a document, and a bag of words, 
    return a features map representing whether or not the 
    document contains each word in the bag of words 
    
    Arguments:
        document:        Words from a document to create feature of 
        bagOfWords:      Words that are to be detected in document 
    '''
    
    documentWords = set(document)
    features = dict()
    for w in documentWords:
        features[w] = (w in bagOfWords)
    return features

def runMNBClassifier(training):
    '''
    Runs and returns Multinomial NaiveBayes on training dataset
    
    Arguments:
        training:        Training data 
    '''        
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training) 
    return MNB_classifier

def runBernoulliClassifier(training):
    '''
    Runs and returns Bernoulli NaiveBayes on training dataset
    
    Arguments:
        training:        Training data 
    '''        
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training) 
    return BernoulliNB_classifier

def runNaiveBayesClassifier(training):
    '''
    Runs and returns NaiveBayesClassifier on training dataset
    
    Arguments:
        training:        Training data 
    '''    
    classifier = nltk.NaiveBayesClassifier.train(training)
    return classifier

def runLogisticRegression(training):
    '''
    Runs and returns classifier based on logistic regression on training dataset
    
    Arguments:
        training:        Training data 
    '''    
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training)
    return LogisticRegression_classifier

def runSGD(training):
    '''
    Runs and returns classifier based on SGD of linear models on 
    training dataset
    
    Arguments:
        training:        Training data 
    '''    
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training)    
    return SGDClassifier_classifier

def runSVC(training):
    '''
    Runs and returns classifier, rbf kernel based SVC on training dataset
    
    Arguments:
        training:        Training data 
    '''    
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training)   
    return SVC_classifier

def runNuSVC(training):
    '''
    Runs and returns classifier based on nu SVC on training dataset
    
    Arguments:
        training:        Training data 
    '''   
    
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training)
    return NuSVC_classifier

def runDecisionTree(training):
    '''
    Runs and returns classifier based on decision trees on training dataset
    
    Arguments:
        training:        Training data 
    '''   
    tree_classifier = SklearnClassifier(DecisionTreeClassifier())
    tree_classifier.train(training)
    return tree_classifier

def runLinearSVC(training):
    '''
    Runs and returns classifier based on linear SVC on training dataset
    
    Arguments:
        training:        Training data 
    '''      
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training)
    return LinearSVC_classifier
    
def returnAccuracy(classifier, testing):
    '''
    Given a trained NaiveBayesClassifier, return accuracy on testing set
    
    Arguments:
        classifier:        Trained classifier
        testing:           Testing dataset
    '''        
    return nltk.classify.accuracy(classifier, testing)

def main():
    allReviews = loadReviews()
    allWords = returnAllWords()
    freq = returnFreqMapOfWords(allWords)
    numFeatures = 3000
    topWords = returnTopWords(freq, numFeatures)
    allFeatures = [(returnFeatureForDocument(lst, topWords), category) for 
                   (lst,category) in allReviews]
    percentageTest = 0.2
    training_set, testing_set = loadTrainingTesting(allFeatures, percentageTest)
    classifier = runNaiveBayesClassifier(training_set)
    
    
    print("Original Naive Bayes Algo accuracy percent:", (returnAccuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(10)
    saveClassifier(classifier, "originalNaiveBayes")
    
    MNB_classifier = runMNBClassifier(training_set)
    print("MNB_classifier accuracy percent:", (returnAccuracy(MNB_classifier, testing_set))*100)
    saveClassifier(MNB_classifier, "MNB")
    
    BernoulliNB_classifier = runBernoulliClassifier(training_set)
    print("BernoulliNB_classifier accuracy percent:", (returnAccuracy(BernoulliNB_classifier, testing_set))*100)
    saveClassifier(BernoulliNB_classifier, "BernoulliNB")
    
    LogisticRegression_classifier = runLogisticRegression(training_set)
    print("LogisticRegression_classifier accuracy percent:", (returnAccuracy(LogisticRegression_classifier, testing_set))*100)
    saveClassifier(LogisticRegression_classifier, "LogisticRegression")
    
    SGDClassifier_classifier = runSGD(training_set)
    print("SGDClassifier_classifier accuracy percent:", (returnAccuracy(SGDClassifier_classifier, testing_set))*100)
    saveClassifier(SGDClassifier_classifier, "SGD")
    
    SVC_classifier =runSVC(training_set)
    print("SVC_classifier accuracy percent:", (returnAccuracy(SVC_classifier, testing_set))*100)
    saveClassifier(SVC_classifier, "SVC")
    
    LinearSVC_classifier = runLinearSVC(training_set)
    print("LinearSVC_classifier accuracy percent:", (returnAccuracy(LinearSVC_classifier, testing_set))*100)
    saveClassifier(LinearSVC_classifier, "LinearSVC")
    
    NuSVC_classifier = runNuSVC(training_set)
    print("NuSVC_classifier accuracy percent:", (returnAccuracy(NuSVC_classifier, testing_set))*100)    
    saveClassifier(NuSVC_classifier, "NuSVC")
    
    tree_classifier = runDecisionTree(training_set)
    print("tree_classifier accuracy percent:", (returnAccuracy(tree_classifier, testing_set))*100) 
    saveClassifier(tree_classifier, "DecisionTree")
    
    listOfClassifierNames = ["NuSVC", "LinearSVC", "SGD", "LogisticRegression",
                             "originalNaiveBayes", "MNB", "BernoulliNB"]
    listOfFinalClassifiers = []
    
    for name in listOfClassifierNames:
        listOfFinalClassifiers.append(loadClassifier(name))
        

    
    voted_classifier = VoteClassifier(listOfFinalClassifiers)
    print("voted_classifier accuracy percent:", (returnAccuracy(voted_classifier, testing_set))*100)
    
main()

