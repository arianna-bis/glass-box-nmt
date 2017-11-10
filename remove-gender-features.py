import argparse
import sys
import re
import FeaturesCombiner

parser = argparse.ArgumentParser(description='map all gender-marked words to the same gender (default is masculine)')

parser.add_argument('--language', required=True, type=str,
                    help='available languages: english, italian')

parser.add_argument('--gender', default='m', type=str,
                    help='gender to which all gender-marked words should be mapped')
parser.add_argument('--lexicon', required=False, type=str,
                    help='path to lefff-like lexicon, format: wordTABposTABlemmaTABfeats')

parser.add_argument('--input', required=False, type=str,
                    help='input filename')

#parser.add_argument('--tagged', required=True, type=str,
#                    help='file containing the tree-tagged corpus in factored format: word|pos|lemma ...')
#parser.add_argument('--out', required=True, type=str,
#                    help='file to print output corpus')
#parser.add_argument('--lexicon', required=True, type=str,
#                    help='path to lefff lexicon, format: wordTABposTABlemmaTABfeats')

config = parser.parse_args()

if config.gender == 'f':
    sys.stderr.write('mapping all gender-marked words to feminine\n')
elif config.gender == 'm':
    sys.stderr.write('mapping all gender-marked words to masculine\n')

f2m_map = {}
f2m_map['una|DET'] = 'un'
f2m_map['un\'|DET'] = 'un'
f2m_map['la|DET'] = 'il'
f2m_map['le|DET'] = 'i'
f2m_map['della|PRE'] = 'del'
f2m_map['delle|PRE'] = 'dei'
f2m_map['dalla|PRE'] = 'dal'
f2m_map['dalle|PRE'] = 'dai'
f2m_map['alla|PRE'] = 'al'
f2m_map['alle|PRE'] = 'ai'
f2m_map['nella|PRE'] = 'nel'
f2m_map['nelle|PRE'] = 'nei'
f2m_map['sulla|PRE'] = 'sul'
f2m_map['sulle|PRE'] = 'sui'
f2m_map['questa|PRO'] = 'questo'
f2m_map['queste|PRO'] = 'questi'
f2m_map['quella|PRO'] = 'quello'
f2m_map['quelle|PRO'] = 'quelli'
f2m_map['mia|PRO'] = 'mio'
f2m_map['mie|PRO'] = 'miei'
f2m_map['sua|PRO'] = 'suo'
f2m_map['sue|PRO'] = 'suoi'
f2m_map['nostra|PRO'] = 'nostro'
f2m_map['nostre|PRO'] = 'nostri'
f2m_map['vostra|PRO'] = 'vostro'
f2m_map['vostre|PRO'] = 'vostri'
f2m_map['lei|PRO'] = 'lui'
f2m_map['ella|PRO'] = 'egli'
f2m_map['esse|PRO'] = 'essi'
f2m_map['essa|PRO'] = 'esso'
f2m_map['la|PRO'] = 'lo'
f2m_map['le|PRO'] = 'li'
f2m_map['glielo|PRO'] = 'gliela'
f2m_map['colei|PRO'] = 'colui'
f2m_map['costei|PRO'] = 'costui'
f2m_map['entrambe|PRO'] = 'entrambi'
f2m_map['ciascuna|PRO'] = 'ciascuno'
f2m_map['ciascune|PRO'] = 'ciascuni'
f2m_map['ognuna|PRO'] = 'ognuno'
f2m_map['nessuna|PRO'] = 'nessuno'
f2m_map['nessune|PRO'] = 'nessuni'
f2m_map['parecchia|PRO'] = 'parecchio'
f2m_map['parecchie|PRO'] = 'parecchi'
f2m_map['alcuna|PRO'] = 'alcuno'
f2m_map['alcune|PRO'] = 'alcuni'
f2m_map['tutta|PRO'] = 'tutto'
f2m_map['tutte|PRO'] = 'tutti'
f2m_map['molta|PRO'] = 'molto'
f2m_map['molte|PRO'] = 'molti'
f2m_map['poca|PRO'] = 'poco'
f2m_map['poche|PRO'] = 'pochi'
f2m_map['troppa|PRO'] = 'troppo'
f2m_map['troppe|PRO'] = 'troppi'
f2m_map['altra|PRO'] = 'altro'
f2m_map['altre|PRO'] = 'altri'

if config.gender == 'f':
    sys.exit('not implemented')

def change_gender_closed_class(word,pos,map):
    word_pos = word + '|' + pos
    if word_pos in map:
        return map[word_pos]
    return word

if config.language == 'english':
    for line in sys.stdin:
        if config.gender == 'm':
            line = re.sub(r'\bshe\b', 'he', line)
            line = re.sub(r'\bher\b', 'his', line)          # THIS IS HACKY!! (could be his or him)
            line = re.sub(r'\bherself\b', 'himself', line)
        else:
            line = re.sub(r'\bhe\b',  'she', line)
            line = re.sub(r'\bhim\b', 'her', line)
            line = re.sub(r'\bhis\b', 'her', line)
            line = re.sub(r'\bhimself\b', 'herself', line)
        sys.stdout.write(line)

elif config.language == 'italian':
    if not config.lexicon:
        sys.exit('must provide lexicon!')

    sys.stderr.write('italian: expecting tree-tagger tagged text as input')

    f = open(config.input, 'r')

    trim_lemma = 0
    fc = FeaturesCombiner.Combiner(config,trim_lemma)
    for line in f:
    #for line in sys.stdin:
        tokens = line.rstrip().split(" ")
        line_out = ""
        for tok in tokens:
            # there can be more than 1 lemma in case of ambiguous words,
            # if so, only the first is kept
            # if len(tok.split("|"))>3:
            #    print("strange: " + tok)
            (word,pos,lem) = tok.split("|",2)
            word = change_gender_closed_class(word,pos,f2m_map)
            word = fc.change_gender(word,pos,config.gender)

            line_out += word+" "
            line = line_out

        sys.stdout.write(line.rstrip() + "\n")

    sys.stderr.write('TODO')
else:
    sys.exit('unknown language')
