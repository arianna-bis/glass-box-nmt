################################################################################
# File: readMonolingEmb.py
# Last edited: 19-5-2017
# By: Clara Tump
#
# this file is run by logregEmbeddings.py before training/testing
# this file reads the train, CV and test sets from .txt files 

################################################################################

from __future__ import print_function
from __future__ import print_function
import time
import sys
import random


# main function: gets the train/cross validation/test set with
# the specified size(number of examples)
def getTCTSets(size1, size2, size3, lang, feature):
    print("lang = " + lang)
    "reading train/cv/test embeddings"
    [trainSet, crossValSet, testSet, trainWords, crossValWords, testWords] = readTCTsets(size1, size2, size3, lang)
    [trainSet, trainLabels, trainWords] = getLabels(trainWords, trainSet, feature)
    # print len(trainWords)
    # for i in range(8):
    # 	print "word: " + trainWords[i]
    # 	print "emb first 3: " + str(trainSet[i][:3])
    # 	print "label: " + trainLabels[i]
    [crossValSet, crossValLabels, crossValWords] = getLabels(crossValWords, crossValSet, feature)
    [testSet, testLabels, finalTestWords] = getLabels(testWords, testSet, feature)
    return [trainSet, crossValSet, testSet, trainLabels, crossValLabels, testLabels, trainWords, finalTestWords]


def getWords(size1, size2, size3, lang):
    print("getting words")
    [trainSet, crossValSet, testSet, trainWords, crossValWords, testWords] = readTCTsets(size1, size2, size3, lang)
    [trainSet, trainLabels, finalTrainWords] = getLabels(trainWords, trainSet, feature)
    [crossValSet, crossValLabels, finalCrossValWords] = getLabels(crossValWords, crossValSet, feature)
    [testSet, testLabels, finalTestWords] = getLabels(testWords, testSet, feature)
    return [finalTrainWords, finalCrossValWords, finalTestWords]


# reads the train/cross validation/test set from word embedding files
def readTCTsets(size1, size2, size3, lang):
    if lang == 'fr2itNew':
        embFile = 'srcEmbeddings.fr2it.ptardis0602.txt'
    elif lang == 'fr2it':
        embFile = 'srcEmbeddings.fr2it.txt'
    elif lang == 'fr2en':
        embFile = 'srcEmbeddings.fr2en.txt'
    elif lang == 'fr2de':
        embFile = 'srcEmbeddings.fr2de.txt'
    else:
        print("embeddings file not available for this language pair")
        sys.exit()
    with open(embFile) as f:
        f.readline()  # firstline is not an embedding
        allSets = []
        totalSize = size1 + size2 + size3
        [allSets, allWords] = addEmbs(allSets, totalSize, f)
        [trainSet, trainWords] = getSets(size1, totalSize, allSets, allWords)
        totalSize = totalSize - size1
        [crossValSet, crossValWords] = getSets(size2, totalSize, allSets, allWords)
        totalSize = totalSize - size2
        [testSet, testWords] = getSets(size3, totalSize, allSets, allWords)
    return [trainSet, crossValSet, testSet, trainWords, crossValWords, testWords]


# adds the specified number of embeddings to the specified set
def addEmbs(embeddings, NoExamples, f):
    words = []
    for i in range(NoExamples):
        line = f.readline()
        if line == '':
            print("end of file")
        else:
            lineList = line.split()
            try:
                words.append(lineList[0])
                del lineList[0]
            except IndexError:
                print(line)
            lineList = map(float, lineList)
            embeddings.append(lineList)
    return [embeddings, words]


def getSets(setSize, totalSize, allSets, allWords):
    xSet = []
    xWords = []
    indices = random.sample(range(0, totalSize), setSize)
    for index in indices:
        xSet.append(allSets[index])
        xWords.append(allWords[index])
    # print("setSize=" + str(setSize) + " totalSize=" + str(totalSize) + "xSet:" + str(xSet[-1]) + "xWords:" + str(xWords[-1]))
    for index in sorted(indices, reverse=True):
        del allSets[index]
        del allWords[index]
    return [xSet, xWords]


####################### labels ########################################
def getLabels(embWords, embeddings, feature):
    wordsAndLabels = []
    file = 'NO-FEATURE'
    if feature == 'gender':
        file = 'wordPlusGenders.txt'
    elif feature == 'number':
        file = 'wordPlusNumbers.txt'
    with open(file) as flabels:
        print("using file: " + file)
        lines = flabels.readlines()
        for line in lines:
            # splitLine = line.split()
            splitLine = line.split(" ", 1)
            word = splitLine[1].rstrip()
            # print "word: " + word
            label = splitLine[0]
            # print "label: " + label
            wordsAndLabels.append((word, label))

    # commonWordsInfo: (word, label, i)
    commonWordsInfo = getCommonWords(wordsAndLabels, embWords)
    finalWords = []
    finalEmbs = []
    finalLabels = []
    finalWords = []
    for wordInfo in commonWordsInfo:
        word = wordInfo[0]
        embIndex = wordInfo[2]
        finalLabel = wordInfo[1]
        finalLabels.append(finalLabel)
        finalEmb = embeddings[embIndex]
        finalEmbs.append(finalEmb)
        finalWords.append(word)
    return [finalEmbs, finalLabels, finalWords]


def getCommonWords(wordsAndLabels, embWords):
    labelWords = []
    for wordAndLabel in wordsAndLabels:
        labelWords.append(wordAndLabel[0])
    commonWordsInfo = []
    for i in range(len(embWords)):
        word = embWords[i]
        if word in labelWords:
            index = labelWords.index(word)
            label = wordsAndLabels[index][1]
            commonWordsInfo.append((word, label, i))
    return commonWordsInfo
