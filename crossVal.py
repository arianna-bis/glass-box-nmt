################################################################################
# 
# File: crossVal.py
# Last edited: 19-5-2017
# By: Clara Tump
#
################################################################################
#
# runs crossValidation by running logregEmbeddings
# multiple times with different values of C (inverse regularization)
# and plotting the accuracies on the Cross Validation set

################################################################################


from __future__ import division
from logregEmbeddings import train
from logregEmbeddings import crossValTest
from logregEmbeddings import getData
from logregEmbeddings import test
import matplotlib.pyplot as plt
import cPickle as pickle
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn import linear_model, datasets
import sys

# total number of embeddings: 30 000
size1 = 10000 #24000
size2 = 3000 #3000
size3 = 3000 #3000
alg = 'skip'
# Change embeddings to use. Options: 'fr2it, fr2en, fr2de'
lang = 'fr2it'
# Feature to predict
feature = 'gender'
#feature = 'number'
	
def makeGraph(Cres):
	plt.figure(1)
	plt.title('FR_IT embeddings: Gender prediction', fontsize=24)
	plt.xlabel('Regularization parameter (C)', fontsize = 20)
	plt.ylabel('accuracy trainset(blue) and testset(red)', fontsize = 18)
	plt.plot(CRes[0], CRes[1], 'ro')
	plt.plot(CRes[0], CRes[2], 'bo')
	plt.tick_params(axis='both', which='major', labelsize=18)
	plt.tick_params(axis='both', which='minor', labelsize=18)
	#plt.xscale('log')
	plt.show()

# load pickled data files previously made by the file readMonolingEmbeddings2
def loadPickledFiles(size1,size2,size3, lang):
	print "loading pickle files"
	X = pickle.load(open("pickle/X" + str(size1) + "-" + str(size2) + "-" + 
								str(size3) + lang + ".pickle", "rb"))
	crossValX = pickle.load(open("pickle/crossValX"+ str(size1) + "-" + 
								str(size2) + "-" + str(size3) + lang + ".pickle", "rb"))
	X2 = pickle.load(open("pickle/X2"+ str(size1) + "-" + str(size2) + "-" +
		 						str(size3) + lang + ".pickle", "rb"))
	XLabels = pickle.load(open("pickle/XLabels"+ str(size1) + "-" + 
								str(size2) + "-" + str(size3) + lang + ".pickle", "rb"))
	crossValLabels = pickle.load(open("pickle/crossValLabels"+ str(size1) + "-" + 
								str(size2) + "-" + str(size3) + lang + ".pickle", "rb"))
	X2Labels = pickle.load(open("pickle/X2Labels"+ str(size1) + "-" + 
								str(size2) + "-" + str(size3) + lang + ".pickle", "rb"))
	testWords = pickle.load(open("pickle/testWords"+ str(size1) + "-" + 
								str(size2) + "-" + str(size3) + lang + ".pickle", "rb"))
	return [X, crossValX, X2, XLabels, crossValLabels, X2Labels, testWords]

# when data is read from readMonolingEmbeddings2 the first time, save them in a pickle
def createPickledFiles(X, crossValX, X2, XLabels, crossValLabels, X2Labels, testWords, lang):
	print "creating pickle files"
	pickle.dump(X, open( "pickle/X" + str(size1) + "-" + str(size2) + "-" + 
							str(size3) + lang + ".pickle", "wb" ) )
	pickle.dump(crossValX, open( "pickle/crossValX"+ str(size1) + "-" + 
							str(size2) + "-" + str(size3) + lang + ".pickle", "wb" ) )
	pickle.dump(X2, open( "pickle/X2"+ str(size1) + "-" + str(size2) + "-" + 
							str(size3) + lang + ".pickle", "wb" ) )
	pickle.dump(XLabels, open( "pickle/XLabels"+ str(size1) + "-" + str(size2) + "-" + 
							str(size3) + lang + ".pickle", "wb" ) )
	pickle.dump(crossValLabels, open( "pickle/crossValLabels"+ str(size1) + "-" + 
							str(size2) + "-" + str(size3) + lang + ".pickle", "wb" ) )
	pickle.dump(X2Labels, open( "pickle/X2Labels"+ str(size1) + "-" + str(size2) + "-" + 
							str(size3) + lang + ".pickle", "wb" ) )
	pickle.dump(testWords, open( "pickle/testWords"+ str(size1) + "-" + str(size2) + "-" + 
							str(size3) + lang + ".pickle", "wb" ) )

def crossValidate(Clist, CRes):
	for i in range(len(Clist)):
		C = Clist[i]
		[logreg, trainAcc] = train(C, 100, 1e-4, X, XLabels)
		accCV = crossValTest(logreg, crossValX, crossValLabels)
		CRes[1].append(accCV)
		CRes[2].append(trainAcc)
		#print str(time.time() - start) + "seconds"
		print("C:" + str(C) + " validAcc:" + str(accCV))
		#sys.stdout.write("C:%f validAcc:%f \n" % (C, accCV))
	makeGraph(CRes)

	# test with best value of C
	max1 = max(CRes[1])
	indices = [i for i, j in enumerate(CRes[1]) if j == max1]
	bestC = Clist[indices[0]]
	return bestC

if __name__ == "__main__":
	start = time.time()
	try:
		[X, crossValX, X2, XLabels, crossValLabels, X2Labels, trainWords, testWords] = loadPickledFiles(size1,size2,size3, lang)
	except (OSError, IOError) as e:
		print "no pickle files available"
		[X, crossValX, X2, XLabels, crossValLabels, X2Labels, trainWords, testWords] = getData(size1,size2,size3,lang,feature)
		createPickledFiles(X, crossValX, X2, XLabels, crossValLabels, X2Labels, testWords, testWords, lang)
	print str(time.time() - start) + "seconds"
	# the values for C which will be tested in cross validation
	#Clist = [1,10,30,50]
	Clist = [0.0001] #,1]
	CRes = [Clist,[],[]]

	bestC = crossValidate(Clist, CRes)
	print str(time.time() - start) + "seconds"
	print "bestC: " + str(bestC)
	[bestLogreg, trainAcc] = train(bestC, 100,1e-4, X, XLabels) 
	testAcc = test(bestLogreg, X2, X2Labels, testWords)
	print "testset accuracy: " + testAcc
	print str(time.time() - start) + "seconds"
