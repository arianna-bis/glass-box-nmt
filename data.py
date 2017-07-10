import gzip
import numpy as np
import sys

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class Data(object):

    """Data loader."""
    def __init__(self, config, fold_suff):

        if not config.label_dict and not config.train_tags:
            sys.exit("Labels must be provided by --label_dict or --train_tags")

        if config.train_tags and not config.labels:
            sys.exit("You must specify which labels to predict when using --train_tags")

        self.label_dict = None
        if config.label_dict:
            self.label_dict = self.load_labels(config.label_dict)

        self.only_tags = None
        if config.only_tags:
            if not config.train_tags:
                sys.exit("option --only_tags requires that tags are provided with the {train|valid|test}_tags options")
            # convert array to hash
            self.only_tags = dict((key, 1) for key in config.only_tags)
            print("using only words with tags:" + " ".join(self.only_tags.keys()))

        if config.labels:
            # convert array to hash
            self.labels = dict((key, 1) for key in config.labels)
            print("predicting labels:" + " ".join(self.labels.keys()))

        self.train_dict = None
        if config.train_dict:
            self.train_dict = self.load_dict(config.train_dict, fold_suff)
        self.train_tags = None
        if config.train_tags:
            self.train_tags = self.load_tags(config.train_tags)
        self.train_words, self.train_vectors, self.train_labels, self.train_size = self.load_vectors_and_labels(
            config.train_vectors, config.n_vectors, self.label_dict, self.train_dict, self.only_tags, self.train_tags)

        self.valid_dict = None
        if config.valid_dict:
            self.valid_dict = self.load_dict(config.valid_dict, fold_suff)
        self.valid_tags = None
        if config.valid_tags:
            self.valid_tags = self.load_tags(config.valid_tags)
        self.valid_words, self.valid_vectors, self.valid_labels, self.valid_size = self.load_vectors_and_labels(
            config.valid_vectors, config.n_vectors, self.label_dict, self.valid_dict, self.only_tags, self.valid_tags)

        self.test_dict = None
        if config.test_dict:
            self.test_dict = self.load_dict(config.test_dict, fold_suff)
        self.test_tags = None
        if config.test_tags:
            self.test_tags = self.load_tags(config.test_tags)
        self.test_words, self.test_vectors, self.test_labels, self.test_size = self.load_vectors_and_labels(
            config.test_vectors, config.n_vectors, self.label_dict, self.test_dict, self.only_tags, self.test_tags)

        del self.label_dict


    # load (at most max_vectors) word vectors from a file
    # also load the label of each word or skip the word if no label is found
    # skip vectors of words that are not found in filter_dict (if provided)
    def load_vectors_and_labels(self, filename, max_vectors, label_dict, filter_dict, only_tags, tags_and_labels):
        print("loading vectors from " + filename + " ...")
        f = fopen(filename, 'r')
        n_features = 0
        vectors = []
        words = []
        labels = []
        n_lines = 0
        n_vectors = 0
        for line in f:
            n_lines += 1
            line = line.rstrip()
            if line.startswith("EOS"):
                continue
            fields = line.split()
            word = fields[0]
            label = None
            if filter_dict and word not in filter_dict:
                continue

            if label_dict:
                if word in label_dict:
                    label = label_dict[word]
            elif tags_and_labels:
                # sanity check:
                if word != tags_and_labels[n_lines-1][0]:
                    sys.exit("word mismatch at line " + str(n_lines-1) + " : " + word + "!=" + tags_and_labels[n_lines-1][0])
                if not only_tags or tags_and_labels[n_lines-1][1] in only_tags:
                    if label != "":
                       label = tags_and_labels[n_lines-1][2]

            if label:
                labels.append(label)
                words.append(word)
                vector = map(float, fields[1:])
                if n_vectors == 0:
                    n_features = len(vector)
                else:
                    assert(n_features==len(vector))
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
    # loads into hash (use for word types)
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

    # expected format: word SPACE tag SPACE lemma SPACE labels
    # loads word,tags,labels into 2-dim array (use for tokens)
    def load_tags(self, filename):
        f = fopen(filename, 'r')
        tokens_tags = []
        n = 0
        for line in f:
            line = line.rstrip()
            if line == "EOS":
                tokens_tags.append(["EOS","_","_","_"])
            else:
                (word,tag,lem,all_labels) = line.split(" ", 3)
                sel_labels = ""
                for l in list(all_labels):
                    if l in self.labels:
                        sel_labels += l
                tokens_tags.append((word,tag,sel_labels))
            n += 1
        print("read " + str(n) + " tokens_tags lines from " + filename)
        print("tokens_tags contains " + str(len(tokens_tags)) + " entries")
        return tokens_tags

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

