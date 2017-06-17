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

parser.add_argument('--alphas', default=None, nargs='+', type=float,
                    help='regularization parameter values to validate \
                          (corresponding to C for LR, alpha for MLP)')
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
parser.add_argument('--n_folds', default=1, type=int,
                    help='number of data folds (expecting files: \
                        train_dict.fold1 ... train_dict.foldN \
                        valid_dict.fold1 ... valid_dict.foldN ...)')

#parser.add_argument('--langpair', required=True, default=None, type=str,
#                    help='language pair code (e.g. fr2it)')

config = parser.parse_args()
print("config:")
print(config)

def run_baseline(config, datasets):
    #model = dummy.DummyClassifier(strategy='stratified')
    model = dummy.DummyClassifier(strategy='most_frequent')
    model.fit(datasets.train_vectors, datasets.train_labels)
    test_acc = model.score(datasets.train_vectors, datasets.train_labels)
    print("baseline(most_freq) test_acc: %.4f" % (test_acc))


def train(config, datasets, alpha_value):
    model = None
    print("using classifier: " + config.classifier)
    if config.classifier == 'LR':
        if not alpha_value:
            alpha_value = 1.0
        model = linear_model.LogisticRegression(verbose=True,solver='lbfgs',C=alpha_value)
    elif config.classifier == 'MLP':
        if not alpha_value:
            alpha_value = 0.0001
        model = neural_network.MLPClassifier(verbose=True,hidden_layer_sizes=(config.mlp_hid,),
                                             activation='relu',solver='adam', alpha=alpha_value)
    model.fit(datasets.train_vectors, datasets.train_labels)
    train_acc = model.score(datasets.train_vectors, datasets.train_labels)
    print("train_acc: : %.4f [alpha=%.4f]" % (train_acc, alpha_value))
    valid_acc = model.score(datasets.valid_vectors, datasets.valid_labels)
    print("valid_acc: : %.4f" % (valid_acc))
    return model, valid_acc

def test(config, datasets, classifier):
    test_acc = None
    if config.preds_file:
        f = open(config.preds_file, 'w')
        preds = classifier.predict(datasets.test_vectors)
        probs = classifier.predict_proba(datasets.test_vectors)
        test_acc = accuracy_score(datasets.test_labels, preds)
        for i in range(len(datasets.test_vectors)):
            prob = '%.4f' % (max(probs[i]))
            f.write(datasets.test_words[i] + "\t" + datasets.test_labels[i] + "\t" + preds[i] + "\t" + prob + "\n")
    else:
        test_acc = classifier.score(datasets.test_vectors, datasets.test_labels)
    print("test_acc: %.4f" % (test_acc))
    return test_acc


n_folds = config.n_folds
datasets = [None] * n_folds
scores = [None] * n_folds
for i in range(n_folds):
    suff_fold = None
    if n_folds > 1:
        suff_fold = '.fold'+str(i+1)    # filename suffixes are 1-numbered
    datasets[i] = data.Data(config, suff_fold)

    classifier = None
    if config.alphas:
        n_classifiers = len(config.alphas)
        classifiers = [None] * n_classifiers
        valid_accs = [None] * n_classifiers
        for j in range(n_classifiers):
            classifiers[j], valid_accs[j] = train(config, datasets[i], config.alphas[j])
        best = np.argmax(valid_accs)
        print("best_alpha: %.4f" % (config.alphas[best]))
        classifier = classifiers[best]
    else:
        classifier, valid_acc = train(config, datasets[i], None)

    scores[i] = test(config, datasets[i], classifier)

print("***\navg_test_acc: %.4f\n***" % (np.mean(scores)))

run_baseline(config,datasets[i])
