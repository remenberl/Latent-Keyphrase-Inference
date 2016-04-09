import argparse
# import re
import sys
sys.path.append('src/tools/')
from filelen import file_len
from myword2vec import Word2Vec
import inflect

parser = argparse.ArgumentParser()
parser.add_argument("-input", help="input path for parsed_text.txt")
parser.add_argument("-stopwords", help="input path for stopwords file")
parser.add_argument("-word2vec", help="input path for word2vec model")
parser.add_argument("-noise", help="input path for noisy domain keyphrases")
parser.add_argument("-window", help="window size")
parser.add_argument("-merge", help="whether to merge plural/singular complete/abbreviated domain keyphrases")
parser.add_argument("-vocab", help="output path for vocabulary")
parser.add_argument("-em", help="output path for em training data")
parser.add_argument("-merged", help="output path for merged domain keyphrases")
args = parser.parse_args()

window_size = int(args.window)
domain_keyphrase_start_id = 100000000

stopwords = set()
with open(args.stopwords, 'r') as input:
    for line in input:
        stopwords.add(line.strip())

noisy_domain_keyphrases = set()
with open(args.noise, 'r') as input:
    for line in input:
        noisy_domain_keyphrases.add(line.split(',')[0].replace('_', ' '))

content_units = dict()
domain_keyphrase_freq = dict()
training_data = list()

total_lines = file_len(args.input)
print "Loading parsed_text.txt:"
with open(args.input, 'r') as input:
    old_percents = 0
    count = 0.0
    for line in input:
        count += 1
        new_percents = count / total_lines
        if new_percents - old_percents > 0.001:
            sys.stdout.write("\rFinish %s percents" % str(new_percents * 100))
            sys.stdout.flush()
            old_percents = new_percents
        seen_content_units = set()
        line = line.replace('$', ' ').strip().lower()
        if line == "":
            continue
        units = []
        domain_keyphrase_indices = []
        current_unit = ''
        within_domain_keyphrase = False
        for ch in line:
            if within_domain_keyphrase == False and ch == " ":
                if current_unit != '':
                    units.append(current_unit)
                    current_unit = ''
            elif ch == '[':
                within_domain_keyphrase = True
            elif ch == ']':
                if current_unit not in noisy_domain_keyphrases:
                    domain_keyphrase_indices.append(len(units))
                within_domain_keyphrase = False
            else:
                current_unit += ch
        if current_unit != '':
            units.append(current_unit)
        pool = []
        right = min(len(units), window_size)
        for index in domain_keyphrase_indices:
            elements = units[index].split(' ')
            if elements[0] in stopwords and elements[-1] in stopwords:
                continue
            if units[index] not in domain_keyphrase_freq:
                domain_keyphrase_freq[units[index]] = 1
            else:
                domain_keyphrase_freq[units[index]] += 1
            if index - right > 0 and len(pool) > 0:
                left = max(0, pool[0] - window_size)
                right = min(len(units), pool[-1] + window_size)
                for i in range(left, right):
                    if units[i] in stopwords:
                        continue
                    if units[i] not in seen_content_units:
                        if units[i] in content_units:
                            content_units[units[i]] += 1
                        else:
                            content_units[units[i]] = 1
                    seen_content_units.add(units[i])
                for i in range(len(pool)):
                    pool[i] -= left
                training_data.append((units[left:right], pool))
                pool = []
            pool.append(index)
            right = min(len(units), index + window_size)
        if len(pool) > 0:
            left = max(0, pool[0] - window_size)
            right = min(len(units), pool[-1] + window_size)
            for i in range(left, right):
                if units[i] in stopwords:
                    continue
                if units[i] not in seen_content_units:
                    if units[i] in content_units:
                        content_units[units[i]] += 1
                    else:
                        content_units[units[i]] = 1
                seen_content_units.add(units[i])
            for i in range(len(pool)):
                pool[i] -= left
            training_data.append((units[left:right], pool))
print

from math import log
content_units_to_id = dict()
content_units_weights = dict()
max_freq = max(content_units.itervalues())
id = 0
with open(args.vocab, 'w') as output:
    output.write('id\tstring\tweight\n')
    for (content_unit, freq) in content_units.iteritems():
        content_units_to_id[content_unit] = domain_keyphrase_start_id + id
        id += 1
        output.write(str(content_units_to_id[content_unit]))
        output.write('\t')
        output.write(content_unit)
        output.write('\t')
        content_units_weights[content_unit] = (-log(float(freq)/max_freq))**2
        output.write(str(content_units_weights[content_unit]))
        output.write('\n')

domain_keyphrase_to_id = dict()
sorted_domain_keyphrase_freq = sorted(domain_keyphrase_freq.items(), key=lambda x:-x[1])

if bool(args.merge):
    tmp_id = 1
    engine = inflect.engine()
    id_to_domain_keyphrases = dict()
    word2vec = Word2Vec(args.word2vec)
    # merge plural and singular phrases
    for domain_keyphrase, freq in sorted_domain_keyphrase_freq:
        words = domain_keyphrase.split(' ')
        if len(words) > 1:
            new_last_word = engine.singular_noun(words[-1])
            if not new_last_word:
                new_last_word = words[-1]
            singular_domain_keyphrase = ' '.join(words[:-1]) + ' ' + new_last_word
            singular_domain_keyphrase = singular_domain_keyphrase.strip()
            if singular_domain_keyphrase not in domain_keyphrase_to_id:
                domain_keyphrase_to_id[singular_domain_keyphrase] = tmp_id
                id_to_domain_keyphrases[tmp_id] = [domain_keyphrase]
                tmp_id += 1
            else:
                if word2vec.cosine(singular_domain_keyphrase, domain_keyphrase) > 0.4:
                    id_to_domain_keyphrases[domain_keyphrase_to_id[singular_domain_keyphrase]].append(domain_keyphrase)
                else:
                    domain_keyphrase_to_id[domain_keyphrase] = tmp_id
                    id_to_domain_keyphrases[tmp_id] = [domain_keyphrase]
                    tmp_id += 1
    # merge domain keyphrases with abbreviations
    unigram_to_id = dict()
    id_to_unigrams = dict()
    for id, domain_keyphrases in id_to_domain_keyphrases.iteritems():
        words = domain_keyphrases[0].split(' ')
        abbrev = ''.join([word[0] for word in words])
        if abbrev in domain_keyphrase_freq and word2vec.cosine(abbrev, domain_keyphrases[0]) > 0.4:
            # id_to_domain_keyphrases[id].append(abbrev)
            unigram_to_id[abbrev] = id
            id_to_unigrams[id] = [abbrev]
    # merge domain keyphrases with nospace versions
    for id, domain_keyphrases in id_to_domain_keyphrases.iteritems():
        words = domain_keyphrases[0].split(' ')
        nospace = ''.join([word for word in words])
        if nospace in domain_keyphrase_freq and word2vec.cosine(nospace, domain_keyphrases[0]) > 0.4:
            id_to_domain_keyphrases[id].append(nospace)
            unigram_to_id[nospace] = id
            id_to_unigrams[id] = [nospace]
    for domain_keyphrase, freq in sorted_domain_keyphrase_freq:
        if len(domain_keyphrase.split(' ')) == 1:
            if domain_keyphrase in unigram_to_id:
                continue
            singular_keyphrase = engine.singular_noun(domain_keyphrase)
            if not singular_keyphrase:
                singular_keyphrase = domain_keyphrase
            if singular_keyphrase != domain_keyphrase and \
                    singular_keyphrase in unigram_to_id and \
                    word2vec.cosine(singular_keyphrase, domain_keyphrase) > 0.4:
                id_to_unigrams[unigram_to_id[singular_keyphrase]].append(domain_keyphrase)
                continue

            if singular_keyphrase not in domain_keyphrase_to_id:
                domain_keyphrase_to_id[singular_domain_keyphrase] = tmp_id
                id_to_domain_keyphrases[tmp_id] = [domain_keyphrase]
                tmp_id += 1
            else:
                if word2vec.cosine(domain_keyphrase_to_id, domain_keyphrase) > 0.4:
                    id_to_domain_keyphrases[domain_keyphrase_to_id[domain_keyphrase_to_id]].append(domain_keyphrase)
                # id_to_domain_keyphrases[unigram_to_id[singular_domain_keyphrase]].append(domain_keyphrase)
                else:
                    domain_keyphrase_to_id[domain_keyphrase] = tmp_id
                    id_to_domain_keyphrases[tmp_id] = [domain_keyphrase]
                    tmp_id += 1
    with open(args.merged, 'w') as output:
        for (tmp_id, domain_keyphrases) in id_to_domain_keyphrases.iteritems():
            size = len(domain_keyphrases)
            if tmp_id in id_to_unigrams:
                size += len(id_to_unigrams[tmp_id])
            if size >= 2:
                for domain_keyphrase in domain_keyphrases:
                    output.write(domain_keyphrase)
                    output.write('\t')
                if tmp_id in id_to_unigrams:
                    for unigram in id_to_unigrams[tmp_id]:
                        if unigram not in domain_keyphrases:
                            output.write(unigram + '\t')
                output.write('\n')
    # reassign id to merged domain_keyphrases
    tmp_id_to_real_id = dict()
    domain_keyphrase_freq_merged = dict()
    for (domain_keyphrase, tmp_id) in domain_keyphrase_to_id.iteritems():
        domain_keyphrase_freq_merged[domain_keyphrase] = sum([domain_keyphrase_freq[i] for i in id_to_domain_keyphrases[tmp_id]])
    sorted_domain_keyphrase_freq = sorted(domain_keyphrase_freq_merged.items(), key=lambda x: -x[1])
    id = 1
    for domain_keyphrase, freq in sorted_domain_keyphrase_freq:
        tmp_id = domain_keyphrase_to_id[domain_keyphrase]
        score = 0
        if len(id_to_domain_keyphrases[tmp_id]) > 1:
            score += 1
        if tmp_id in id_to_unigrams:
            score += 1
        elements = id_to_domain_keyphrases[tmp_id][0].split(' ')
        if score < 1 and (elements[0] in stopwords or elements[-1] in stopwords):
            continue
        tmp_id_to_real_id[tmp_id] = id
        id += 1
    total_freq = sum(domain_keyphrase_freq_merged.values())
    with open(args.vocab, 'a') as output:
        for (domain_keyphrase, tmp_id) in domain_keyphrase_to_id.iteritems():
            if tmp_id not in tmp_id_to_real_id:
                continue
            output.write(str(tmp_id_to_real_id[tmp_id]))
            output.write('\t')

            max_freq = 0
            rep = ''
            for unit in id_to_domain_keyphrases[tmp_id]:
                if domain_keyphrase_freq[unit] > max_freq:
                    rep = unit
                    max_freq = domain_keyphrase_freq[unit]
            output.write(rep)
            output.write('\t')
            freq = domain_keyphrase_freq_merged[domain_keyphrase]
            output.write(str(float(freq) / total_freq + 0.000001))
            output.write('\n')
    domain_keyphrase_to_id.clear()
    for (tmp_id, domain_keyphrases) in id_to_domain_keyphrases.iteritems():
        for domain_keyphrase in domain_keyphrases:
            try:
                if tmp_id in tmp_id_to_real_id:
                    domain_keyphrase_to_id[domain_keyphrase] = tmp_id_to_real_id[tmp_id]
            except:
                continue
else:
    id = 1
    for domain_keyphrase, freq in sorted_domain_keyphrase_freq:
        domain_keyphrase_to_id[domain_keyphrase] = id
        id += 1
    total_freq = sum(domain_keyphrase_freq.values())
    with open(args.vocab, 'a') as output:
        for (domain_keyphrase, id) in domain_keyphrase_to_id.iteritems():
            output.write(str(id))
            output.write('\t')
            output.write(domain_keyphrase)
            output.write('\t')
            output.write(str(float(domain_keyphrase_freq[domain_keyphrase]) / total_freq + 0.000001))
            output.write('\n')

print "Writing training data."
with open(args.em, 'w') as output:
    for (content_units, domain_keyphrase_indices) in training_data:
        size = 0
        for index in domain_keyphrase_indices:
            if content_units[index] in domain_keyphrase_to_id:
                size += 1
        if size > 0:
            pool = set()
            for unit in content_units:
                if unit in content_units_to_id:
                    output.write(str(content_units_to_id[unit]))
                    output.write(' ')
            for index in domain_keyphrase_indices:
                if content_units[index] in domain_keyphrase_to_id and \
                        domain_keyphrase_to_id[content_units[index]] not in pool:
                    output.write(str(domain_keyphrase_to_id[content_units[index]]))
                    output.write(' ')
                    pool.add(domain_keyphrase_to_id[content_units[index]])
            output.write('\n')
