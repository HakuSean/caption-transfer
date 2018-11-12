# from the extracted attributes, get the map
# For later usage, will include this part into the attribute classifier.

import h5py
import numpy as np
import torch as t
import argparse
import os
import csv
import time

parser = argparse.ArgumentParser(description="mAP")
parser.add_argument('--input_gt', default='dataset_coco.h5', type=str, 
                    help='input path and name of the ground truth')
parser.add_argument('--input_file', default='attributes.h5', type=str, 
                    help='input path and name of the attributes file')
parser.add_argument('--no_stopwords', action='store_true', default=False)

args = parser.parse_args()


def main():
    start = time.time()
    gt = h5py.File(args.input_gt)
    att = h5py.File(args.input_file)
    assert len(att) == len(gt), "The input files need to be the same length, and the keys should be strictly the same."

    gt_array = []
    att_array = []

    for key in gt.keys():
        gt_array.append(gt[key][:])
        att_array.append(att[key][:])

    print('Read data:', time.time() - start)

    gt_array = np.array(gt_array)
    att_array = np.array(att_array)

    precision, recall = epoch_map(gt_array, att_array)
    #print('The overall precision is %.4f.' % precision)

    # ap and mAP
    ap = []
    for c in range(len(gt_array[0])):
        p = precision[c]
        r = recall[c]
        ap.append(compute_ap(p, r))

    ap = np.array(ap)
    mAP = ap.mean()

    print('Computing:', time.time() - start)

    # attributes
    if args.no_stopwords:
        f = open('../attributes.txt', 'r')
    else:
        f = open('../attributes_with_stopwords.txt', 'r')
    attribute_names = f.readlines()
    attribute_names = [x.strip() for x in attribute_names]
    f.close()

    # printout
    # for i in range(len(ap)):
        # print('AP for '+attribute_names[i] + ' = ' + str(ap[i]))

    csvname = './attributes/results_' + os.path.basename(args.input_gt).split('.')[0] + '/' + os.path.basename(args.input_file) + '.csv'
    with open(csvname, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(ap)):
            writer.writerow([attribute_names[i], ap[i]])

    print('The mAP for the file '+ os.path.basename(args.input_file) + ' is ' + str(mAP))

def epoch_map(labels, logits, delta=1e-11):
    precision = []
    recall = []

    # loop over all classes
    for c in range(len(labels[0])):
        label = labels[:, c]
        logit = logits[:, c]
        prec_c = []
        rec_c = []
        index = np.argsort(- logit) # descending
        for i in range(1, len(label)+1):
            gt_chose = label[index[:i]]
            gt_unchose = label[index[i:]]
            tp =  gt_chose.sum()
            fp = (1 - gt_chose).sum()
            fn = gt_unchose.sum()
            prec_c.append(tp / (tp + fp + delta))
            rec_c.append(tp / (tp + fn + delta))
        precision.append(prec_c)
        recall.append(rec_c)

    return precision, recall

def compute_ap(prec, rec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':

    main()