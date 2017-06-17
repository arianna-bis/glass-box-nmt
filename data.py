import gzip
import numpy as np


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class Data(object):

    """Data loader."""
    def __init__(self, config, fold_suff):
        self.label_dict = self.load_labels(config.label_dict)

        self.train_dict = None
        if config.train_dict:
            self.train_dict = self.load_dict(config.train_dict, fold_suff)
        self.train_words, self.train_vectors, self.train_labels, self.train_size = self.load_vectors_and_labels(
            config.train_vectors, config.n_vectors, self.label_dict, self.train_dict)

        self.valid_dict = None
        if config.valid_dict:
            self.valid_dict = self.load_dict(config.valid_dict, fold_suff)
        self.valid_words, self.valid_vectors, self.valid_labels, self.valid_size = self.load_vectors_and_labels(
            config.valid_vectors, config.n_vectors, self.label_dict, self.valid_dict)

        self.test_dict = None
        if config.test_dict:
            self.test_dict = self.load_dict(config.test_dict, fold_suff)
        self.test_words, self.test_vectors, self.test_labels, self.test_size = self.load_vectors_and_labels(
            config.test_vectors, config.n_vectors, self.label_dict, self.test_dict)

        del self.label_dict

    # load (at most max_vectors) word vectors from a file
    # also load the label of each word or skip the word if no label is found
    # skip vectors of words that are not found in filter_dict (if provided)
    def load_vectors_and_labels(self, filename, max_vectors, label_dict, filter_dict):
        f = fopen(filename, 'r')
        vectors = []
        words = []
        labels = []
        n_lines = 0
        n_vectors = 0
        for line in f:
            n_lines += 1
            fields = line.split()
            word = fields[0]
            if word not in label_dict:
                continue
            if filter_dict and word not in filter_dict:
                continue
            labels.append(label_dict[word])
            words.append(word)
            vector = map(float, fields[1:])
            vectors.append(vector)
            n_vectors+=1
            if n_vectors == max_vectors:
                break

        print("read " + str(n_lines) + " vector lines from " + filename)
        print("loaded " + str(n_vectors) + " vectors")

        vectors = np.array(vectors)
        labels = np.transpose(np.array(labels))
        return words, vectors, labels, n_vectors

    # expected format: label SPACE word
    def load_labels(self, filename):
        f = fopen(filename, 'r')
        label_dict = {}
        n = 0
        for line in f:
            fields = line.rstrip().split(" ", 1)
            label_dict[fields[1]] = fields[0]
            n += 1
        print("read " + str(n) + " label_dict lines from " + filename)
        print("label_dict contains " + str(len(label_dict)) + " entries")
        return label_dict

    # expected format: one entry per line
    def load_dict(self, filename, fold_suff):
        if fold_suff:
            filename = filename + fold_suff
        f = fopen(filename, 'r')
        dict = {}
        n = 0
        for line in f:
            dict[line.rstrip()] = 1
            n += 1
        print("read " + str(n) + " dict lines from " + filename)
        print("dict contains " + str(len(dict)) + " entries")
        return dict

