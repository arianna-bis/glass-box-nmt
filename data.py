import gzip
import numpy as np
import sys

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def freq_to_freqbin(freq):
    maxFreqs = [100,500,1000,2000,3000,5000]
    for maxFreq in maxFreqs:
        if freq < maxFreq:
            return str(maxFreq)
    return '>' + str(maxFreqs[-1])

class Data(object):

    """Data loader."""
    def __init__(self, config, fold_suff):

        self.unk = "<unk>"
        self.eos = "EOS"


        if not config.label_dict and not config.train_tags:
            sys.exit("Labels must be provided by --label_dict or --train_tags")

        if config.train_tags and not config.labels:
            sys.exit("You must specify which labels to predict when using --train_tags")

        self.freqbin_dict = None
        if config.freq_dict:
            self.freqbin_dict = self.load_freqbins(config.freq_dict)

        self.label_dict = None
        if config.label_dict:
            self.label_dict = self.load_labels(config.label_dict)

        self.only_tags = None
        if config.only_tags == ['all'] or config.only_tags == ['ALL']:
            config.only_tags = None
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
            if line.startswith(self.eos) or line.startswith(self.unk):
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
                # append the tag to the word, for analysis purposes:
                word += '__' + tags_and_labels[n_lines-1][1]
                # filter the sample by tag, if needed:
                if not only_tags or tags_and_labels[n_lines-1][1] in only_tags:
                    if label != "":
                        label = tags_and_labels[n_lines-1][2]

            if label:
                labels.append(label)
                words.append(word)
                vector = list(map(float, fields[1:]))
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

    # expected format: word TAB frequency
    # bin frequencies and load into hash (use for word types)
    def load_freqbins(self, filename):
        f = fopen(filename, 'r')
        freqbin_dict = {}
        n = 0
        for line in f:
            fields = line.rstrip().split("\t", 1)
            freqbin_dict[fields[0]] = freq_to_freqbin(int(fields[1]))
            n += 1
        print("read " + str(n) + " freq_dict lines from " + filename)
        print("freq_dict contains " + str(len(freqbin_dict)) + " entries")
        return freqbin_dict

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
            if line == self.eos:
                tokens_tags.append([self.eos,"_","_","_"])
            else:
                (word,tag,lem,all_labels) = line.split(" ", 3)
                #sel_labels = ""
                selected_labels = {}

                for l in list(all_labels):
                    if l in self.labels:
                        #sel_labels += l
                        #assert(len(sel_labels)==1)  # make sure only one feature value per word is retained
                        selected_labels[l] = 1

                selected_labels_str = ''
                if(len(selected_labels.keys())==1):
                    selected_labels_str = ''.join(selected_labels.keys())

                tokens_tags.append((word,tag,selected_labels_str))

                #if tag == 'VER':
                #    print('\t'.join((word,tag,selected_labels_str)))
                #    print(''.join(all_labels))
                #    print('')

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

