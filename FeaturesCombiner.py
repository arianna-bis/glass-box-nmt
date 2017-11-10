import sys
import re


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class Combiner(object):

    """Data loader."""
    def __init__(self, config, trim_lemma=0):
        self.config = config

        # treetagger-to-lefff tag correspondences
        pmap = {}
        if self.config.language == 'french':
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

        elif self.config.language == 'italian':
            pmap['VER']=['ver','aux','mod','cau']
            pmap['ADJ']=['adj']
            pmap['NOM']=['noun']
            pmap['NPR']=['npr']
            pmap['ADV']=['adv']
            pmap['PRO']=['pro']
            pmap['DET']=['det']
            pmap['ART']=['det']
            pmap['PRE']=['pre','artpre']
            pmap['PON']=['pon']
            pmap['SENT']=['sent']

        self.pos_map = pmap
        self.lexicon,self.lex_lpm_word = self.load_lefff(config.lexicon,trim_lemma)

    # expected format: wordTABposTABlemmaTABfeats
    # loads into hash where KEY=word|pos VALUE=(lemma,feats)
    def load_lefff(self,filename,trim_lemma=0):
        f = fopen(filename, 'r')
        lex_by_wordpos = {}
        lex_by_lemmaposmfeats = {}
        n = 0
        n_duplicate_keys = 0
        for line in f:
            fields = line.rstrip().split("\t")
            #(word,pos,lemma,sfeats) = line.rstrip().split("\t")
            word = fields[0]
            pos = fields[1]
            lemma = fields[2]
            if trim_lemma>0 and len(lemma)>trim_lemma:
                lemma = lemma[:-trim_lemma]
            sfeats = ""
            if len(fields) > 3:
                sfeats = fields[3]
            #feats = sfeats.split("")
            key = word + "|" + pos  # + "|" + lemma
            value = (lemma,sfeats)

            if key in lex_by_wordpos:
                #print('duplicate word+pos+lem key! ' + key)
                n_duplicate_keys += 1
            lex_by_wordpos[key] = value

            key = lemma + "|" + pos + "|" + sfeats
            value = word
            lex_by_lemmaposmfeats[key] = word

            n += 1
        sys.stderr.write("read " + str(n) + " lefff lines from " + filename + "\n")
        sys.stderr.write("lexicon contains " + str(len(lex_by_wordpos)) + " entries\n")
        sys.stderr.write("lexicon contains " + str(n_duplicate_keys) + " duplicate keys\n")
        return lex_by_wordpos, lex_by_lemmaposmfeats

    def get_morphfeats(self,word,ttag_pos,ttag_lem):
        if ttag_pos in self.pos_map:
            for lefff_pos in self.pos_map[ttag_pos]:
                word_pos = word + "|" + lefff_pos
                if word_pos in self.lexicon:
                    morphfeats = self.lexicon[word_pos][1]
                    if morphfeats == "":
                        morphfeats = "__"
                    # if multiple POS tags correspond to this word,
                    # only the first provided in the mapping is used
                    # to get the morph. features
                    return morphfeats
        return "__"

    def change_gender(self,word,ttag_pos,gender):
        changed_word = word
        if ttag_pos in self.pos_map:
            for lefff_pos in self.pos_map[ttag_pos]:
                word_pos = word + "|" + lefff_pos
                if word_pos in self.lexicon:
                    lemma = self.lexicon[word_pos][0]
                    morphfeats = self.lexicon[word_pos][1]
                    if self.config.language == 'french':
                        morphfeats = '-'.join(morphfeats.split(""))
                    new_feats = ''
                    if gender == 'f':
                        new_feats = re.sub(r'\bm\b','f',morphfeats)
                    else:
                        new_feats = re.sub(r'\bf\b','m',morphfeats)

                    key = lemma + '|' + lefff_pos + '|' + new_feats
                    if key in self.lex_lpm_word:
                        changed_word = self.lex_lpm_word[key]
                        return changed_word
        return changed_word
