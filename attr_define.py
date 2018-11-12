## Count number of words and calculate the relationship
## the input should be the json which has the same format of dataset_coco.json

import numpy as np
import argparse
from nltk.corpus import wordnet
import re
import itertools
import os
import json, h5py
import operator

parser = argparse.ArgumentParser()
parser.add_argument('sent_files', nargs='+', type=str,
                    help='only two inputs, one for source, the other for target')
parser.add_argument('--without_frequency', action='store_true', default=False)
parser.add_argument('--keep_stopwords', action='store_true', default=False)
parser.add_argument('--threshold', type=int, default=256, help='the top X of the most frequent words')
args = parser.parse_args()

def parse_json(json_file):
    img = json_file['images']
    parse_dict = {}

    for i in img:
        idx = str(i['cocoid']).encode("utf-8").decode("utf-8")
        parse_dict[idx] = []
        for s in i['sentences']:
            parse_dict[idx].append(s['raw'])
    return parse_dict


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

def count(sent_group, stopwords, keep_stopwords, without_frequency):
    word_count = {}
    word_ind = {}
    ind = 0
    for sents in sent_group:
        for line in sents:
            line = line.lower().strip('\n\r\t\.')
            line = re.sub('[\\/\n\t\.;:!,_&?]', ' ', line)# keep all the inter-dash, and prime 
            words = line.split(' ')
            for word in words:
                if word == '': # remove nothing
                    continue
                word = word.strip('\n\r\t')
                word = word.strip('\'\"\.:!;,_&?-') # if the dash is in head or tail, remove
                word = re.sub(r'[0-9]', '', word)
                word = word_original_form(word)
                if (not keep_stopwords) and stopwords.has_key(word):
                    continue
                if re.sub(r'[^a-z]', '', word) == '': # remove everything that are not letters
                    continue

                if word_count.has_key(word) and not without_frequency:
                    word_count[word] =  word_count[word] + 1
                    word_ind[word].append(ind)
                else:
                    word_count[word] = 1
                    word_ind[word] = [ind] # the ind is the same as order of sent_group and img_list, but is not the same as the cocoids
        ind += 1
    return word_count, word_ind

def compute_attributes(word_count_list, threshold):
    # based on the first list, add the second
    total_count = word_count_list[0]
    if len(word_count_list) == 2: # consider both source and target
        target_list = word_count_list[1]
        # all words must also in target domain
        for word in total_count.keys():
            if not target_list.has_key(word):
                total_count[word] = 0
        # the frequency should be added together
        for word in target_list.keys():
            if total_count.has_key(word):
                total_count[word] += target_list[word]

    sorted_count = sorted(total_count.items(), key=operator.itemgetter(1), reverse=True)
    attributes = dict(sorted_count[0:threshold]).keys() # the output is re-ordered without relation to the frequency
    return attributes

def write_file(img_names, word_index, file_name, attributes):    
    # constitute the labels for each attribute
    labels = np.zeros((len(img_names), len(attributes)))
    for aix in range(len(attributes)):
        word = attributes[aix]
        indeces = word_index[word]
        for ind in indeces:
            labels[ind][aix] = 1.0

    # testify there is no all-zero labels
    label_sum = np.sum(labels, axis=1)
    if 0 in label_sum:
        print('Some images have no labels.')
    else:
        print('All images have labels.') 

    with h5py.File(file_name) as file_label:
        for imgid, label in zip(img_names, labels):
            file_label[imgid] = label
        file_label.close()

# input should be json files
sent_json_files = [json.load(open(x, 'r')) for x in args.sent_files]
parse_dicts = [parse_json(x) for x in sent_json_files]
img_list = [x.keys() for x in parse_dicts]
sent_list = [x.values() for x in parse_dicts] # same len as image, but each item contains 5 to 50 sentences

without_frequency = args.without_frequency
keep_stopwords = args.keep_stopwords

# define stop words
stop_word_reader = open('../vocabulary/stop_word_list.txt', 'r')
stop_word_list = stop_word_reader.readlines()
stop_word_reader.close()
stop_words = {}

for stop_word in stop_word_list:
    stop_words[stop_word.strip('\n\t\r')] = 1

# count words in each corpus
word_count_list = [count(sent_group, stop_words, keep_stopwords, without_frequency) for sent_group in sent_list]
word_index_list = [x[1] for x in word_count_list]
word_count_list = [x[0] for x in word_count_list]
print('Finished counting words.')

# generate attributes
attributes = compute_attributes(word_count_list, args.threshold)
print('Successfully defined attributes')

f = open('../attributes.txt', 'w')
for key in attributes:
    f.write(key+'\n')

f.close()

# generate files for all word_count in word_count_list
output_dir = 'attributes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for cnt, img, name in zip(word_index_list, img_list, args.sent_files):
    output_name = os.path.join(output_dir, os.path.splitext(os.path.basename(name))[0]+'.h5')
    write_file(img, cnt, output_name, attributes)
    print('Files wrtten in hdf5!!')


