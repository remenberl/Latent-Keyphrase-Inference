from nltk.corpus import wordnet as wn
from nltk.corpus.reader import NOUN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-input", help="input path for concepts file")
parser.add_argument("-output", help="output path for noise file")
args = parser.parse_args()

results = list()
with open(args.input, 'r') as input:
    for line in input:
        concept = line.split(',')[0]
        words = concept.split('_')
        word = ''
        if len(words) > 1:
            # results.append(line)
            synsets = wn.synsets(concept)
            if len(synsets) != 0:
                noun_synsets = wn.synsets(concept, NOUN)
                if len(noun_synsets) == 0:
                    results.append(line)
                    continue
            else:
                continue
                word = words[-1]
        else:
            word = concept
        synsets = wn.synsets(word)
        if len(synsets) == 0:
            # concept not in database
            # results.append(line)
            pass
        else:
            noun_synsets = wn.synsets(word, NOUN)
            if len(noun_synsets) == 0:
                results.append(line)

with open(args.output, 'w') as output:
    for line in results:
        output.write(line)
