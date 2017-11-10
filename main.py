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

# Labels from lexicon (e.g. lexicon only containing gender features)
parser.add_argument('--label_dict', default=None, type=str,
                    help='file containing the dictionary of word labels')
# OR
# Labels from tagged corpus
# format per line: word POS lemma feats
parser.add_argument('--train_tags', default=None, type=str,
                    help='file containing the tags and features of the training words')
parser.add_argument('--valid_tags', default=None, type=str,
                    help='file containing the tags and features of the valid words')
parser.add_argument('--test_tags', default=None, type=str,
                    help='file containing the tags and features of the test words')
parser.add_argument('--labels', default=['m','f'], nargs='+', type=str,
                    help='labels (i.e. morph.features) to predict')


parser.add_argument('--only_baseline', action='store_true',
                    help='only compute baseline accuracy')

# Classifier options:
parser.add_argument('--classifier', default='LR', type=str,
                    help='classifier to use: LR or MLP')
parser.add_argument('--alphas', default=None, nargs='+', type=float,
                    help='regularization parameter values to validate \
                          (corresponding to C for LR, alpha for MLP)')
parser.add_argument('--mlp_hid', default=100, type=int,
                    help='hidden layer size for MLP classifier')

# Output files:
parser.add_argument('--preds_file', default=None, type=str,
                    help='file to print the predictions for the test set')

# Data selection options:
parser.add_argument('--train_dict', default=None, type=str,
                    help='file containing the words to filter the training data')
parser.add_argument('--valid_dict', default=None, type=str,
                    help='file containing the words to filter the valid data')
parser.add_argument('--test_dict', default=None, type=str,
                    help='file containing the words to filter the test data')
parser.add_argument('--only_tags', default=None, nargs='+', type=str,
                    help='only keep samples with the provided tags. \
                          Tags must be provided with the {train|valid|test}_tags options')

# Data folds:
parser.add_argument('--n_folds', default=1, type=int,
                    help='number of data folds (expected files: \
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
    #test_acc = model.score(datasets.test_vectors, datasets.test_labels)
    preds = model.predict(datasets.test_vectors)
    test_acc = accuracy_score(datasets.test_labels, preds)
    print("baseline(most_freq) test_acc: %.4f" % (test_acc))

    test_acc_by_tag = {}
    if config.test_tags:
        test_acc_by_tag = test_breakdown_by_tag(datasets.test_labels, preds, datasets.test_words, '__')

    return test_acc, test_acc_by_tag


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

def test(config, datasets, classifier, suff_fold):
    test_acc = None
    preds = classifier.predict(datasets.test_vectors)
    test_acc = accuracy_score(datasets.test_labels, preds)
    print("test_acc: %.4f" % (test_acc))

    if config.preds_file:
        preds_file = config.preds_file
        if suff_fold:
            preds_file = preds_file + suff_fold
        f = open(preds_file, 'w')
        probs = classifier.predict_proba(datasets.test_vectors)
        for i in range(len(datasets.test_vectors)):
            prob = '%.4f' % (max(probs[i]))
            f.write(datasets.test_words[i] + "\t" + datasets.test_labels[i] + "\t" + preds[i] + "\t" + prob + "\n")

    test_acc_by_tag = {}
    if config.test_tags:
        test_acc_by_tag = test_breakdown_by_tag(datasets.test_labels, preds, datasets.test_words, '__')

    return test_acc, test_acc_by_tag


# split the test samples by tags and return an accuracy for each sample subset
def test_breakdown_by_tag(y_true, y_pred, y_wordtags, delim):
    assert(len(y_true)==len(y_pred)==len(y_wordtags))
    subsets = {}
    for i in range(len(y_true)):
        _, tag = y_wordtags[i].split(delim, 1)
        if tag not in subsets.keys():
            # initialize with two empty arrays: one for y_true and one for y_pred:
            subsets[tag] = []
            subsets[tag].append([])
            subsets[tag].append([])

        subsets[tag][0].append(y_true[i])
        subsets[tag][1].append(y_pred[i])

    test_acc_by_tag = {}
    for tag in subsets.keys():
        # for each tag, store the respective accuracy and the number of samples with that tag
        test_acc_by_tag[tag] = (accuracy_score(subsets[tag][0],subsets[tag][1]), len(subsets[tag][0]))

    return test_acc_by_tag


# expects an array of dictionaries: one dictionary per fold contains the accuracies broken down by tag
def print_avg_scores_by_tag(scores_by_tag):
    # gather all accuracies by tag
    avg_scores_by_tag = {}
    for tag in scores_by_tag[0].keys():
        avg_scores_by_tag[tag] = []
        avg_scores_by_tag[tag].append( [scores_by_tag[0][tag][0]] )
        avg_scores_by_tag[tag].append( [scores_by_tag[0][tag][1]] )

    for f in range(1, len(scores_by_tag)):
        for tag in scores_by_tag[f].keys():
            avg_scores_by_tag[tag][0].append(scores_by_tag[f][tag][0])  # append accuracy
            avg_scores_by_tag[tag][1].append(scores_by_tag[f][tag][1])  # append nb samples

    print("avg test_acc breakdown by tag:")
    for tag in sorted(avg_scores_by_tag.keys()):
        #tag_acc = np.mean(avg_scores_by_tag[tag][0])
        print("TAG=" + tag + "\t %.4f [std: %.4f] [avg#samples: %.1f]"
              % (np.mean(avg_scores_by_tag[tag][0]), np.std(avg_scores_by_tag[tag][0]), np.mean(avg_scores_by_tag[tag][1])))

n_folds = config.n_folds
datasets = [None] * n_folds
scores = [0] * n_folds
scores_by_tag = [0] * n_folds
base_scores = [0] * n_folds
base_scores_by_tag = [0] * n_folds
for i in range(n_folds):

    suff_fold = None
    if n_folds > 1:
        suff_fold = '.fold'+str(i+1)    # filename suffixes are 1-numbered
    datasets[i] = data.Data(config, suff_fold)
    base_scores[i], base_scores_by_tag[i] = run_baseline(config, datasets[i])
    if not config.only_baseline:
        classifier = None
        if config.alphas:
            n_classifiers = len(config.alphas)
            classifiers = [None] * n_classifiers
            valid_accs = [None] * n_classifiers
            for j in range(n_classifiers):
                classifiers[j], valid_accs[j] = train(config, datasets[i], config.alphas[j])
            best = np.argmax(valid_accs)
            print("best_alpha: %.4f" % (config.alphas[best]))
            classifier = classifiers[int(best)]
        else:
            classifier, valid_acc = train(config, datasets[i], None)
        scores[i], scores_by_tag[i] = test(config, datasets[i], classifier, suff_fold)

print("*** avg baseline test_acc: %.4f [std: %.4f] ***" % (np.mean(base_scores), np.std(base_scores)))
print("*** avg classif  test_acc: %.4f [std: %.4f] ***" % (np.mean(scores),      np.std(scores)))

print("\nBASELINE SCORES BY TAG:")
if config.test_tags:
    print_avg_scores_by_tag(base_scores_by_tag)

print("\nCLASSIFIER SCORES BY TAG:")
if config.test_tags:
    print_avg_scores_by_tag(scores_by_tag)

