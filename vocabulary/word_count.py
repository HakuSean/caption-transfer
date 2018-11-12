## Count number of words and calculate the relationship

import numpy as np
import argparse
from nltk.corpus import wordnet
import re
import itertools
import os

parser = argparse.ArgumentParser()
parser.add_argument('sent_files', nargs='+', type=str)
parser.add_argument('--with_frequency', action='store_true', default=False)
parser.add_argument('--remove_stopwords', action='store_true', default=False)
args = parser.parse_args()

def word_original_form(word):
    temp = wordnet.morphy(word, wordnet.NOUN)
    if word == temp or temp is None:
       temp = wordnet.morphy(word, wordnet.VERB)
       if word == temp or temp is None:
           temp = wordnet.morphy(word, wordnet.ADJ)
           if temp is None:
               temp = word
    if temp is not None:
        word = str(temp)
    return word

def count(sent_group, stopwords, remove_stopwords, with_frequency):
    word_count = {}
    for sents in sent_group:
        for line in sents:
            line = line.lower().strip('.:!,_&? ')
            line = re.sub('\.(?!$)', ' ', line) # keep all the inter-dash
            words = line.split(' ')
            for word in words:
                word = re.sub(r'[0-9]', '', word)
                word = word_original_form(word)
                if remove_stopwords and stopwords.has_key(word):
                    continue
                if re.sub(r'[^a-z]', '', word) == '': # remove everything that are not letters
                    continue

                if word_count.has_key(word) and with_frequency:
                    word_count[word] =  word_count[word] + 1
                else:
                    word_count[word] = 1
    return word_count

def intersection(word_count1, word_count2):
    cnt = 0.0
    for key in word_count1.keys():
        if word_count2.has_key(key):
            cnt += word_count1[key]
    return cnt/sum(word_count1.values())

def vocab_relation(word_count_list):
    relation = {}
    for pair, index in zip(itertools.permutations(word_count_list, 2), itertools.permutations(sent_files, 2)): # pick up any two to form a pair and omit the diagonal
        relation[str(index)+'/'+index[0]] = intersection(pair[0], pair[1])
    return relation

sent_files = [os.path.basename(x) for x in args.sent_files]
sent_npz_files = [np.load(x) for x in args.sent_files]
img_list = [x['image'] for x in sent_npz_files]
sent_list = [x['sent'] for x in sent_npz_files] # same len as image, but each item contains 5 to 50 sentences

with_frequency = args.with_frequency
remove_stopwords = args.remove_stopwords

# define stop words
stop_word_reader = open('stop_word_list.txt', 'r')
stop_word_list = stop_word_reader.readlines()
stop_word_reader.close()
stop_words = {}
for stop_word in stop_word_list:
    stop_words[stop_word.strip('\n\t\r')] = 1

# count words in each corpus
word_count_list = [count(sent_group, stop_words, remove_stopwords, with_frequency) for sent_group in sent_list]

# vocabulary relationships between any two corpuses
matrix = vocab_relation(word_count_list)

# print and save files
print 'The vocabulary statistics are as below:'
print matrix



