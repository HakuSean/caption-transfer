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
import random

parser = argparse.ArgumentParser()
parser.add_argument('sent_files', nargs='+', type=str,
                    help='only two inputs, one for source, the other for target')
parser.add_argument('--without_frequency', action='store_true', default=False)
parser.add_argument('--keep_stopwords', action='store_true', default=False)
parser.add_argument('--threshold', type=int, default=256, help='the top X of the most frequent words')
parser.add_argument('--max_len', type=int, default=30, help='maximum length of the label. 40 for with stopwords (average 18 in pascal), 30 when without (9.9 in pascal).')
args = parser.parse_args()

def parse_json(json_file):
    img_list = json_file['images']
    sents = []
    imgid = []

    for i in img_list:
        idx = str(i['cocoid']).encode("utf-8").decode("utf-8")
        imgid.append(idx)
        sents.append([])
        for s in i['sentences']:
            sents[-1].append(s['raw'])
    return imgid, sents


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
        # the frequency should be added together ???
        for word in target_list.keys():
            if total_count.has_key(word):
                total_count[word] += target_list[word]

    sorted_count = sorted(total_count.items(), key=operator.itemgetter(1), reverse=True) # a list of tuples: ('white', 686)
    attributes = dict(sorted_count[0:threshold]) # the output is re-ordered without relation to the frequency, the frequency is kept
    return attributes

def write_with_order(img_names, word_index, file_name, attributes):    
    # constitute the labels for each attribute
    all_attributes = attributes.keys()
    random.seed(1000)
    random.shuffle(all_attributes)

    labels = np.zeros((len(img_names), len(all_attributes)))
    for aix in range(len(all_attributes)):
        word = all_attributes[aix]
        indeces = word_index[word]
        for ind in indeces:
            labels[ind][aix] = 1.0

    # testify there is no all-zero labels
    label_sum = np.sum(labels, axis=1)
    if 0 in label_sum:
        print('Some images have no labels.')
        #print(max(label_sum))
        #print(label_sum.mean())
    else:
        print(max(label_sum))
        print(label_sum.mean())
        print('All images have labels.') 

    ix_to_word = {str(i):v for i,v in enumerate(all_attributes, 1)}
    ix_to_word['0'] = 'UNK'

    # write labels in order
    labels_with_order = []
    label_length = []
    for instance in labels:
        indices = [i for i, j in enumerate(instance) if j == 1.] 
        cnts = [attributes[all_attributes[i]] for i in indices] # the frequency of each attribute, used for sorting
        
        sorted_index = sorted(range(len(cnts)), key=lambda k: cnts[k], reverse=True)
        label = [indices[i] for i in sorted_index]
        if len(indices) >= args.max_len:
            label = label[:args.max_len]
            label_len = args.max_len # the length of labels
        else:
            label += [0]* (args.max_len - len(indices))
            label_len = len(indices)
        
        labels_with_order.append(label)
        label_length.append(label_len)

    with h5py.File(file_name+'.h5', 'w') as file_label:
        file_label.create_dataset("labels", dtype='uint32', data=labels_with_order)
        file_label.create_dataset("label_length", dtype='uint32', data=label_length)
        file_label.close()

    with h5py.File(file_name+'_gts.h5', 'w') as file_label:
        for imgid, label in zip(img_names, labels):
            file_label[imgid] = label
        file_label.close()

    # create output json file
    out = {}
    out['ix_to_word'] = ix_to_word # encode the (1-indexed) vocab, kind of a dictionary
    out['images'] = []
    for i,img in enumerate(img_names):
    
        jimg = {}
        jimg['id'] = img # copy over & mantain an id, if present (e.g. coco ids, useful)
        out['images'].append(jimg)
  
    json.dump(out, open(file_name+'.json', 'w'))


# input should be json files
sent_json_files = [json.load(open(x, 'r')) for x in args.sent_files]
img_list = []
sent_list = []
for x in sent_json_files:
    imgs, sents = parse_json(x)
    img_list.append(imgs)
    sent_list.append(sents)

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

# generate files for all word_count in word_count_list
if args.keep_stopwords:
    output_dir = 'attributes_gt_with_step_with_stopwords'
else:
    output_dir = 'attributes_gt_with_step_nostopwords'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for cnt, img, name in zip(word_index_list, img_list, args.sent_files):
    output_name = os.path.join(output_dir, os.path.splitext(os.path.basename(name))[0])
    write_with_order(img, cnt, output_name, attributes)
    print('Files wrtten in hdf5!!')


