from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import h5py

def language_eval(dataset, preds, model_id, split):
    import sys
    if 'coco' in dataset:
        annFile = '../data/coco/annotations/captions_val2014.json'
    elif 'pascal' in dataset:
        # sys.path.append("f30k-caption")
        annFile = '../data/pascal/pascal_5S_gts.json'
    elif 'flickr' in dataset:
        annFile = '../data/flickr/annotations/dataset_flickr30k.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    #annFile = dataset
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json') # the json file to store the overall scores, and the caption and scores for each image.

    coco = COCO(annFile) # loading annFile to memory.
    valids = coco.getImgIds() # a list that contains all ids of images, ids are int

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids] # sometimes there are not every images in the preds_filt
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API... # Sean: write in the image_id and predicted caption

    cocoRes = coco.loadRes(cache_path) # evaluation results, actually did not have too much relationship with coco in image captions (have some relation in segmentation tasks)
    cocoEval = COCOEvalCap(coco, cocoRes) 
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate() # the results are generated from this

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval # this is used for output json file
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True) # if has the key, return the value; if not, return True. However, the eval_kwargs itself will not be updated
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1)) # will be an important input in both training and test
    split = eval_kwargs.get('split', 'val') # used for language_eval function only, only for writing files and to distinguish between val and test results
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'pascal') # in validation, this is given
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator() # each time of eval will go over all the test images

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch()
        n = n + loader.batch_size # batch_size is 10, which comes from the original batch size

        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp

            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)

        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq) # from index to words, these are the output sentences.

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption'])) # entry['caption'] is the output caption, eg. image 5616: a man with a bottle of a on top

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
        # lang_stats = language_eval(loader.coco_json, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


def eval_and_probmap(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True) # if has the key, return the value; if not, return True. However, the eval_kwargs itself will not be updated
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1)) # will be an important input in both training and test
    split = eval_kwargs.get('split', 'val') # used for language_eval function only, only for writing files and to distinguish between val and test results
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'pascal') # in validation, this is given
    beam_size = eval_kwargs.get('beam_size', 1)
    dir_weight = eval_kwargs.get('input_fc_dir', -1)
    proc_method = eval_kwargs.get('proc', 'mean')
    seq_length = eval_kwargs.get('seq_length', 16)
    assert dir_weight != -1, 'Did not provide store directory for weights.'
    print('The path to store the generated weights is '+dir_weight)

    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator() # each time of eval will go over all the test images
    file_weights = h5py.File(os.path.join(dir_weight, 'weights_' + proc_method + '.h5'))
    file_mask = h5py.File(os.path.join(dir_weight, 'mask_' + proc_method + '.h5')) # 0, 1 to indicate which words are used.

    n = 0
    predictions = []
    weights = []
    use_all = False
    while True:
        data = loader.get_batch()
        n = n + loader.batch_size # batch_size is 1 for generating the probmap

        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
                    data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                    data['labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels = tmp

            weight = model(fc_feats, att_feats, labels).cpu()
            seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
            #print(weight.shape)

        # get the processed weight
        # use mean
        if proc_method == 'mean':
            print('Using mean for weights.')
            weight_proc = weight.mean(1).squeeze()
        elif proc_method == 'max':
            print('Using max for weights.')
            weight_proc = weight.max(1)[0].squeeze() # use max
            weight_proc = weight_proc/weight_proc.sum() # do not use softmax
            # weight_proc = F.softmax(weight_proc)
        else:
            print('Selecting all weights.')
            use_all = True
            weight_proc = weight.squeeze().data.cpu().float().numpy()
            length, dim = weight_proc.shape
            if length < seq_length:
                weight_proc = np.append(weight_proc, np.zeros((seq_length-length, dim)), axis=0)
                mask = np.array([1.]*length+[0.]*(seq_length-length))
            elif length > seq_length:
                weight_proc = weight_proc[:seq_length, :]
                mask = np.array([1.]*seq_length)
            
        if use_all:
            d_set_weights = file_weights.create_dataset(str(data['infos'][0]['id']), (seq_length, weight.size(-1)), dtype="float")
            d_set_weights[...] = weight_proc
            d_set_mask = file_mask.create_dataset(str(data['infos'][0]['id']), (seq_length, ), dtype="float")
            d_set_mask[...] = mask
        else:
            d_set_weights = file_weights.create_dataset(str(data['infos'][0]['id']), (weight.size(-1), ), dtype="float")
            d_set_weights[...] = weight_proc.data.cpu().float().numpy()

        sents = utils.decode_sequence(loader.get_vocab(), seq) # from index to words, these are the output sentences.

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption'])) # entry['caption'] is the output caption, eg. image 5616: a man with a bottle of a on top

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    file_mask.close()
    file_weights.close()

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
        # lang_stats = language_eval(loader.coco_json, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return predictions, lang_stats