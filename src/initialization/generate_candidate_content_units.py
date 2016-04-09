from collections import defaultdict, Counter
import argparse
import sys
sys.path.append('src/tools/')
from myword2vec import Word2Vec
from filelen import file_len

parser = argparse.ArgumentParser()
parser.add_argument("-em", help="input path for em training data")
parser.add_argument("-merged", help="input path for merged domain keyphrases")
parser.add_argument("-vocab", help="input path for vocabulary")
parser.add_argument("-word2vec", help="input path for word2vec model")
parser.add_argument("-ann", help="input path for most similar word pairs")
parser.add_argument("-child", help="max number of content units for domain keyphrases")
parser.add_argument("-candi", help="output path for candidate content units")
args = parser.parse_args()

domain_keyphrase_start_id = 100000000

vocabulary = dict()
reverse_vocabulary = dict()
freq = defaultdict(int)
conditional_freq = defaultdict(list)
# conditional_freq = dict()
with open(args.vocab, 'r') as input:
    input.readline()
    for line in input:
        elements = line.split('\t')
        vocabulary[int(elements[0])] = elements[1]
        if int(elements[0]) >= domain_keyphrase_start_id:
            reverse_vocabulary[elements[1]] = int(elements[0])

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
        for element in content_units:
            freq[element] += 1
        for domain_keyphrase in domain_keyphrases:
            for content_unit in content_units:
                if domain_keyphrase != content_unit:
                    # conditional_freq[domain_keyphrase][content_unit] += 1
                    conditional_freq[domain_keyphrase].append(content_unit)
print

word2vec = Word2Vec(args.word2vec)

size = len(conditional_freq)
candidate_content_units = dict()
count = 0
old_percents = 0
print "Computing pairwise similarity between domain_keyphrases and content_units:"
# record = open('tmp/freq_compare.txt', 'w')

neighbours = dict()
with open(args.ann, 'r') as input:
    for line in input:
        elements = line.split('\t')
        neighbours[elements[0].strip()] = [elements[1].strip(), elements[3].strip(), elements[5].strip()]

for word in conditional_freq.iterkeys():
    # output.write(word)
    pool = list()
    word_string = vocabulary[word]
    count += 1
    new_percents = float(count) / len(conditional_freq)
    freq_word = freq[reverse_vocabulary[word_string]]
    if new_percents - old_percents > 0.001:
        sys.stdout.write("\rFinish %s percents" % str(new_percents * 100))
        sys.stdout.flush()
        old_percents = new_percents
    norm_sims = []
    if word_string.replace(' ', '_') not in neighbours:
        # print word_string
        continue
    for neighbour in neighbours[word_string.replace(' ', '_')]:
        sim = word2vec.cosine(word_string, neighbour)
        if sim != -1:
            norm_sims.append(sim)
    mean_norm_sim = sum(norm_sims) / len(norm_sims)
    for (candidate, frequency) in dict(Counter(conditional_freq[word])).iteritems():
        candidate_string = vocabulary[candidate]
        # record.write(word_string + '\t' + str(freq_word) + '\t' + candidate_string + '\t' + str(freq[candidate])+ '\t' + str(frequency) + '\n')
        if frequency >= max(0.01 * freq_word, 3):
            sim = word2vec.cosine(word_string, candidate_string) / mean_norm_sim
            if freq[candidate] <= 10 * freq_word and \
                    freq[candidate] / frequency < 30:
                if sim > 0.5 and (sim >= 0.7 or freq[candidate] <= freq_word):
                    pool.append((candidate_string, float(frequency)**1.3 / freq[candidate]))
            elif sim > 0.25 and \
                    float(frequency) / freq_word + sim / 2 > 0.55:
                pool.append((candidate_string, float(frequency)**1.3 / freq[candidate]))
    if len(pool) < 5:
        continue
    pool = sorted(pool, key=lambda x: -x[1])
    candidate_content_units[word_string] = set()
    for (candi, score) in pool[:int(args.child)]:
        candidate_content_units[word_string].add(candi)
print

synonyms = dict()
if args.merged:
    with open(args.merged, 'r') as input:
        for line in input:
            elements = line.strip().split('\t')
            for element1 in elements:
                synonyms[element1] = set()
                for element2 in elements:
                    synonyms[element1].add(element2)

for domain_keyphrase, content_units in candidate_content_units.iteritems():
    all_units = set()
    for content_unit in content_units:
        all_units.add(content_unit)
        if content_unit in synonyms:
            for unit in synonyms[content_unit]:
                all_units.add(unit)
    candidate_content_units[domain_keyphrase] = all_units

for domain_keyphrase, content_units in candidate_content_units.iteritems():
    for content_unit in content_units:
        if content_unit in candidate_content_units:
            candidate_content_units[content_unit].add(domain_keyphrase)

with open(args.candi, 'w') as output:
    for domain_keyphrase, content_units in candidate_content_units.iteritems():
        if len(content_units) > 0:
            output.write(domain_keyphrase)
            all_units = set()
            for content_unit in content_units:
                all_units.add(content_unit)
                if content_unit in synonyms:
                    for unit in synonyms[content_unit]:
                        all_units.add(unit)
            for unit in all_units:
                output.write('\t' + unit)
            output.write('\n')

