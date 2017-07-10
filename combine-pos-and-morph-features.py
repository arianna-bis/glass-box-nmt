import argparse
import gzip
import sys

parser = argparse.ArgumentParser(description='combine POS tags (from tree-tagged file) with morphological features (from lefff lexicon)')

#parser.add_argument('--tagged', required=True, type=str,
#                    help='file containing the tree-tagged corpus in factored format: word|pos|lemma ...')
#parser.add_argument('--out', required=True, type=str,
#                    help='file to print output corpus')
parser.add_argument('--lexicon', required=True, type=str,
                    help='path to lefff lexicon, format: wordTABposTABlemmaTABfeats')

config = parser.parse_args()

# treetagger-to-lefff tag correspondences
pmap = {}
pmap['NOM']=['nc']
pmap['PRP']=['prep']
pmap['VER']=['v','auxAvoir','auxEtre']
pmap['DET']=['det']
pmap['ADJ']=['adj','adjPref']
pmap['PRO']=['pro','cln','cld','cla','clr','cll','pri','cldr']
pmap['PUN']=['poncts','ponctw']
pmap['ADV']=['adv','advPref','advneg']
pmap['KON']=['coo']
pmap['NAM']=['np']

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

# expected format: wordTABposTABlemmaTABfeats
# loads into hash where KEY=word|pos VALUE=(lemma,feats)
def load_lefff(filename):
    f = fopen(filename, 'r')
    lexicon = {}
    n = 0
    for line in f:
        fields = line.rstrip().split("\t")
        #(word,pos,lemma,sfeats) = line.rstrip().split("\t")
        word = fields[0]
        pos = fields[1]
        lemma = fields[2]
        sfeats = ""
        if len(fields) > 3:
            sfeats = fields[3]
        #feats = sfeats.split("")
        key = word + "|" + pos
        value = (lemma,sfeats)
        lexicon[key] = value
        n += 1
    sys.stderr.write("read " + str(n) + " lefff lines from " + filename + "\n")
    sys.stderr.write("lexicon contains " + str(len(lexicon)) + " entries\n")
    return lexicon

def get_morphfeats(word,ttag_pos,lefff,pos_map):
    if ttag_pos in pos_map:
        for lefff_pos in pos_map[ttag_pos]:
            word_pos = word + "|" + lefff_pos
            if word_pos in lefff:
                morphfeats = lefff[word_pos][1]
                if morphfeats == "":
                    morphfeats = "__"
                # if multiple POS tags correspond to this word,
                # only the first provided in the mapping is used
                # to get the morph. features
                return morphfeats
    return "__"

lexicon = load_lefff(config.lexicon)

#f_in = fopen(config.tagged, 'r')
#f_out = open(config.out, 'w')
for line in sys.stdin:
    tokens = line.rstrip().split(" ")
    line_out = ""
    for tok in tokens:
        # there can be more than 1 lemma in case of ambiguous words,
        # if so, only the first is kept
        # if len(tok.split("|"))>3:
        #    print("strange: " + tok)
        (word,pos,lem) = tok.split("|",2)
        feats = get_morphfeats(word,pos,lexicon,pmap)
        line_out += (word+"|"+pos+"|"+lem+"|"+feats+" ")
    sys.stdout.write(line_out.rstrip() + "\n")
