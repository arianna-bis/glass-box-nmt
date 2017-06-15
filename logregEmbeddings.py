################################################################################
# File: logregEmbeddings.py
# Last edited: 18-5-2017
# By: Clara Tump
#
################################################################################
# methods from this file are run by crossVal.py during crossvalidation
# this file handles all training,testing on Cross validation set
# and testing on test set of the pipeline, reports the accuracy scores.
# A logistic regression model is used. 
# Necessary: scikit-learn version 0.18rc2.

################################################################################

from __future__ import print_function
from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
import sklearn
from sklearn import neural_network
from sklearn import datasets
from readMonolingEmb2 import getTCTSets
from sklearn import linear_model


def getData(size1, size2, size3, lang, feature):
    [X, crossValX, X2, XLabels, crossValLabels, X2Labels, trainWords, testWords] = getTCTSets(size1, size2, size3, lang,
                                                                                              feature)
    print("size train set: " + str(len(X)) + " out of " + str(size1))
    print("size crossVal set: " + str(len(crossValX)) + " out of " + str(size2))
    print("size testset: " + str(len(X2)) + " out of " + str(size3))
    X = np.array(X)
    XLabels = np.array(XLabels)
    np.transpose(XLabels)
    return [X, crossValX, X2, XLabels, crossValLabels, X2Labels, trainWords, testWords]


########## TRAIN ###############################################################
def train(C, maxiter, tol, X, XLabels):
    print("10th train emb: " + str(X[10][:3]))
    print("10th trainlabel: " + str(XLabels[10]))
    print("15th train emb: " + str(X[15][:3]))
    print("15th trainlabel: " + str(XLabels[15]))
    print("20th train emb: " + str(X[20][:3]))
    print("20th trainlabel: " + str(XLabels[20]))

    print("using neural network")
    logreg = neural_network.MLPClassifier(verbose=True, hidden_layer_sizes=(100,),
                                          activation='relu',
                                          solver='lbfgs')  # alpha=C, learning_rate_init = 0.0001, beta_1 = 0.9, epsilon = 10e-8)
    # line below would be for training using logistic regression instead of MLP
    # logreg = linear_model.LogisticRegression(tol = tol, max_iter = maxiter,
    # solver='liblinear', C=C)
    logreg.fit(X, XLabels)
    trainAcc = str(logreg.score(X, XLabels))
    print('trainAcc=' + trainAcc)
    return [logreg, trainAcc]


########## TEST ON CROSSVAL ####################################################
def crossValTest(logreg, crossValX, crossValLabels):
    crossValPrediction = logreg.predict(crossValX)

    correct = 0
    for i, j in zip(crossValLabels, crossValPrediction):
        if i == j:
            correct = correct + 1
    testScore = str(correct / len(crossValLabels))
    return testScore


########## TEST ################################################################
def test(logreg, X2, X2Labels, testWords):
    f = open('results.txt', 'w')
    f.write("word/label/predicted/probability\n")
    X2prediction = logreg.predict(X2)
    predicProba = logreg.predict_proba(X2)
    correct = 0
    for i in range(len(X2Labels)):
        label = X2Labels[i]
        predic = X2prediction[i]
        word = testWords[i]
        proba = max(predicProba[i])
        f.write(word + "/" + label + "/" + predic + "/" + str(proba) + "\n")
        if label == predic:
            correct = correct + 1
    testScore = str(correct / len(X2Labels))
    f.close()
    return testScore

# if no cross validation is needed, run logreg from this file
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='run prediction exps')

    parser.add_argument('--vectors', required=True, default=None, type=str,
                        help='file containing the word vectors (types or tokens)')
    parser.add_argument('--labels', required=True, default=None, type=str,
                        help='file containing the dictionary of word labels')
    parser.add_argument('--lang', required=True, default=None, type=str,
                        help='language pair code (e.g. fr2it)')

    args = parser.parse_args()

    size1 = 1000000
    size2 = 100000
    size3 = 100000
    lang = args.lang
    feature = "number"
    [X, crossValX, X2, XLabels, crossValLabels, X2Labels, trainWords, testWords] = getData(size1, size2, size3, lang,
                                                                                           feature)
    [logreg, trainAcc] = train(0.1, 1, 1e-4, X, XLabels)
    testAcc = test(logreg, X2, X2Labels, testWords)
    print ("testAcc:" + testAcc)
# trainAccBIS = test(logreg, X, XLabels, trainWords)
# print ("trainAccBIS:" + trainAccBIS)

