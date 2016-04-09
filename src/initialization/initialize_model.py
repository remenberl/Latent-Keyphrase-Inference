from collections import defaultdict, Counter
from math import log, sqrt
import argparse
import sys
sys.path.append('src/tools/')
from filelen import file_len
from myword2vec import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument("-em", help="input path for em training data")
parser.add_argument("-candi", help="input path for candidate content units")
parser.add_argument("-vocab", help="input path for vocabulary")
parser.add_argument("-word2vec", help="input path for word2vec model")
parser.add_argument("-model", help="output path for model file")
args = parser.parse_args()

domain_keyphrase_start_id = 100000000

conditional_freq = defaultdict(list)
count = 0
total_lines = file_len(args.em)
print "Loading EM training data:"
with open(args.em, 'r') as input:
    old_percents = 0
    count = 0.0
    for line in input:
        count += 1
        new_percents = count / total_lines
        if new_percents - old_percents > 0.001:
            sys.stdout.write("\rFinish %s percents" % str(new_percents * 100))
            sys.stdout.flush()
            old_percents = new_percents
        elements = line.strip().split(' ')
        elements = [int(element) for element in elements]
        domain_keyphrases = []
        content_units = []
        for element in elements:
            if element < domain_keyphrase_start_id:
                domain_keyphrases.append(element)
            else:
                content_units.append(element)
        for domain_keyphrase in domain_keyphrases:
            for content_unit in content_units:
                conditional_freq[domain_keyphrase].append(content_unit)
print

content_units_to_id = dict()
domain_keyphrase_to_id = dict()
domain_keyphrase_prior = dict()
with open(args.vocab, 'r') as input:
    input.readline()
    for line in input:
        elements = line.split('\t')
        if domain_keyphrase_start_id > int(elements[0]):
            domain_keyphrase_to_id[elements[1]] = int(elements[0])
            domain_keyphrase_prior[elements[1]] = float(elements[2])
        else:
            content_units_to_id[elements[1]] = int(elements[0])

domain_keyphrase_to_content_units = dict()
with open(args.candi, 'r') as input:
    for line in input:
        elements = line.split('\t')
        if elements[0].strip() in domain_keyphrase_to_id:
            domain_keyphrase_to_content_units[elements[0].strip()] = set()
            if len(elements) > 1:
                for element in elements:
                    if element.strip() != '':
                        domain_keyphrase_to_content_units[elements[0].strip()].add(element.strip())

word2vec = Word2Vec(args.word2vec)
with open(args.model, 'w') as output:
    output.write('parent\tchild\tweight\n')
    for (domain_keyphrase, connected_content_units) in domain_keyphrase_to_content_units.iteritems():
        # downweight unigram
        if len(domain_keyphrase.split(' ')) == 1:
            domain_keyphrase_prior[domain_keyphrase] = domain_keyphrase_prior[domain_keyphrase] * 0.01
        leak_prob = -log(1 - domain_keyphrase_prior[domain_keyphrase])
        output.write('0\t' + str(domain_keyphrase_to_id[domain_keyphrase]) + '\t' + str(leak_prob) + '\n')
        pool = list()
        freq = dict(Counter(conditional_freq[domain_keyphrase_to_id[domain_keyphrase]]))
        energy = max(freq.itervalues())
        for content_unit in connected_content_units:
            cosine_sim = word2vec.cosine(domain_keyphrase, content_unit)
            if content_units_to_id[content_unit] in freq:
                pool.append((content_unit, sqrt(freq[content_units_to_id[content_unit]] / float(energy)) * cosine_sim))
            else:
                pool.append((content_unit, cosine_sim))
        pool = sorted(pool, key=lambda x: -x[1])
        for (content_unit, sim) in pool:
            output.write(str(domain_keyphrase_to_id[domain_keyphrase]))
            output.write('\t')
            output.write(str(content_units_to_id[content_unit]))
            output.write('\t')
            output.write(str(max(sim, 0.01)))
            output.write('\n')
            if content_unit in domain_keyphrase_to_content_units and domain_keyphrase_to_id[domain_keyphrase] < domain_keyphrase_to_id[content_unit]:
                output.write(str(domain_keyphrase_to_id[domain_keyphrase]))
                output.write('\t')
                output.write(str(domain_keyphrase_to_id[content_unit]))
                output.write('\t')
                output.write(str(max(sim, 0.01)))
                output.write('\n')

    for content_unit in content_units_to_id.iterkeys():
        # leak_prob = -log(1 - min(0.99, max(0.02, 1+(log(float(freq)/max_freq) / 10)  )))* 0.000001
        output.write('0\t' + str(content_units_to_id[content_unit]) + '\t' + str(0.00001) + '\n')
