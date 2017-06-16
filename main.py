import argparse
import numpy as np
import sklearn
from sklearn import neural_network
from sklearn import linear_model
from sklearn import dummy
from sklearn.metrics import accuracy_score
import data


parser = argparse.ArgumentParser(description='run prediction exps')

parser.add_argument('--train_vectors', default='train_vecs', type=str,
                    help='file containing the training word vectors (types or tokens)')
parser.add_argument('--valid_vectors', default='valid_vecs', type=str,
                    help='file containing the valid word vectors (types or tokens)')
parser.add_argument('--test_vectors', default='test_vecs', type=str,
                    help='file containing the test word vectors (types or tokens)')
parser.add_argument('--n_vectors', default=-1, type=int,
                    help='maximum nb of vectors to include in the training set')

parser.add_argument('--mlp_hid', default=100, type=int,
                    help='hidden layer size for MLP classifier')

parser.add_argument('--label_dict', default='label_dict', type=str,
                    help='file containing the dictionary of word labels')
parser.add_argument('--classifier', default='LR', type=str,
                    help='classifier to use: LR or MLP')
parser.add_argument('--preds_file', default=None, type=str,
                    help='file to print the predictions for the test set')

parser.add_argument('--train_dict', default=None, type=str,
                    help='file containing the words to filter the training data')
parser.add_argument('--valid_dict', default=None, type=str,
                    help='file containing the words to filter the valid data')
parser.add_argument('--test_dict', default=None, type=str,
                    help='file containing the words to filter the test data')



#parser.add_argument('--langpair', required=True, default=None, type=str,
#                    help='language pair code (e.g. fr2it)')

config = parser.parse_args()
print("config:")
print(config)

def baseline(config, datasets):
    #model = dummy.DummyClassifier(strategy='stratified')
    model = dummy.DummyClassifier(strategy='most_frequent')
    model.fit(datasets.train_vectors, datasets.train_labels)
    test_acc = model.score(datasets.train_vectors, datasets.train_labels)
    print("baseline(most_freq) test_acc: %.4f" % (test_acc))


def train(config, datasets):
    model = None
    print("using classifier: " + config.classifier)
    if config.classifier == 'LR':
        model = linear_model.LogisticRegression(verbose=True,solver='lbfgs')  # todo C=?
    elif config.classifier == 'MLP':
        model = neural_network.MLPClassifier(verbose=True,hidden_layer_sizes=(config.mlp_hid,),
                                             activation='relu',solver='adam')
    model.fit(datasets.train_vectors, datasets.train_labels)
    train_acc = model.score(datasets.train_vectors, datasets.train_labels)
    print("train_acc: : %.4f" % (train_acc))
    return model

def test(config, datasets, classifier):
    test_acc = None
    if config.preds_file:
        f = open(config.preds_file, 'w')
        preds = classifier.predict(datasets.test_vectors)
        probs = classifier.predict_proba(datasets.test_vectors)
        test_acc = accuracy_score(datasets.test_labels, preds)
        for i in range(len(datasets.test_vectors)):
            #'%s %s' % ('one', 'two')
            #'%f' % (3.141592653589793,)
            prob = '%.4f' % (max(probs[i]))
            f.write(datasets.test_words[i] + "\t" + datasets.test_labels[i] + "\t" + preds[i] + "\t" + prob + "\n")
    else:
        test_acc = classifier.score(datasets.test_vectors, datasets.test_labels)
    print("test_acc: : %.4f" % (test_acc))

mydata = data.Data(config)

classifier = train(config, mydata)
test(config, mydata, classifier)
baseline(config,mydata)