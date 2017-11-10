import argparse
import gzip
import sys
import re
import FeaturesCombiner

parser = argparse.ArgumentParser(description='combine POS tags (from tree-tagged file) with morphological features (from lefff lexicon)')

parser.add_argument('--language', default='french', type=str,
                    help='available languages: french, italian')

#parser.add_argument('--tagged', required=True, type=str,
#                    help='file containing the tree-tagged corpus in factored format: word|pos|lemma ...')
#parser.add_argument('--out', required=True, type=str,
#                    help='file to print output corpus')
parser.add_argument('--lexicon', required=True, type=str,
                    help='path to lefff lexicon, format: wordTABposTABlemmaTABfeats')

config = parser.parse_args()

def main():
    fc = FeaturesCombiner.Combiner(config)
    n_empty_feats = 0
    for line in sys.stdin:
        tokens = line.rstrip().split(" ")
        line_out = ""
        for tok in tokens:
            # there can be more than 1 lemma in case of ambiguous words,
            # if so, only the first is kept
            # if len(tok.split("|"))>3:
            #    print("strange: " + tok)
            (word,pos,lem) = tok.split("|",2)
            feats = fc.get_morphfeats(word,pos,lem)
            if feats == "__":
                n_empty_feats += 1
            line_out += (word+"|"+pos+"|"+lem+"|"+feats+" ")
        sys.stdout.write(line_out.rstrip() + "\n")
    sys.stderr.write("num tokens with no morph features: " + str(n_empty_feats) + "\n")


main()
