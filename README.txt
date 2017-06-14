Files in this folder:
readMonolingEmb2.py		- reads the embeddings&morphological dictionary, and returns labelled train/cv/test set 
logregEmbeddings.py		- trains and tests using logistic regression
crossVal.py				- coordinates everything: training/cross validating / testing with best parameters	

How to run:
python crossVal.py	

if you run crossVal.py, it will now run gender prediction on fr_it embeddings, on all the data, using an MLP, cross validating for the parameter C (regularization). When cross validation is finished it will show you a graph with C on the x-axis and the cross validation accuracy on the y-axis. If you close this graph, it will run the MLP on the test set using the C with the best performance.

TO CHANGE THINGS:
1. Change word embedding files:
- change the value for embFile in readMonolingEmb2 on line 44
3. Change the number of word embeddings used (eg if you only want to test quickly on 2000 embeddings)
- change the size1/size2/size3 on line 97-99 in crossVal.py (size1=trainsize, size2=cvsize, size3=testsize)
3. Change from gender prediction to number prediction:
- change 'file' on line 99 of readMonolingEmb2 from 'wordPlusGenders' to 'wordPlusNumbers'

