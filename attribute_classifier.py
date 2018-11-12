
# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
import json
import pickle
import pprint
import h5py
import math
import numpy as np
import os
import argparse
from random import shuffle
from collections import Counter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset

BATCH_SIZE = 16
INI_DISC_WEIGHT_SCALE = -1
INI_DISC_BIAS = 0.5
LAST_WEIGHT_LIMIT = -2

#################################
#      define the network       #
#################################

#define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, classes)
        #self.fc3 = nn.BatchNorm1d(batchsize)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.2)(x)
        x = self.fc2(x)
        
        #x = nn.Dropout(p=0.4)(x)
        #x=self.fc3(x)
        return x

class Contrast_ReLU_activate(nn.Module):

    def __init__(self, initWeightScale, initBias):

        super(Contrast_ReLU_activate, self).__init__()

        self.dom_func_weight = nn.Parameter(torch.ones(1),requires_grad=True)
        self.dom_func_bias = Variable(torch.FloatTensor([0]).cuda())

        self.weight_scale = initWeightScale
        self.add_bias = initBias

    def forward(self, dom_res, dom_label, init_weight):

        w = (self.dom_func_weight * self.weight_scale).expand_as(init_weight)
        b = (self.dom_func_bias + self.add_bias).expand_as(init_weight)

        dom_prob = F.sigmoid(dom_res).squeeze()
        dom_variance = torch.abs(dom_prob - 0.5)

        act_weight = 0.8 + w * (200 * dom_variance)**4  + b
        
        # Minimise function to zero(target)
        zeros_var = b
        f_weight = torch.max(act_weight, zeros_var)

        final_weight = f_weight
        
        return final_weight, w.squeeze().data[0], b.squeeze().data[0]


########################################
#        define the score              #
########################################

#define the precision, recall and F1 score
def f1_score(label, predict, threshold=0.5, delta=1e-11):
    
    #binarized the label and predicted output
    label = np.array(label)
    predict = np.array(predict)
    label = label>threshold
    predict = predict>threshold

    #define the precision recall and F1-score
    tp = np.sum(np.bitwise_and(label, predict))
    fp = np.sum(np.bitwise_and(np.invert(label), predict))
    fn = np.sum(np.bitwise_and(label, np.invert(predict)))
    precision = float(tp) / (tp + fp + delta)
    recall = float(tp) / (tp + fn + delta)
    F1 = 2.0 * precision * recall / (precision + recall + delta) 
    return precision, recall, F1

def result(testData, testGT):
#input: training data(N*2048) + training label (N*256)
#output: avearage precision, recall and F1 score
    test_Num = len(testData)
    outputs = []
    iteration = int(math.ceil((float(test_Num)/batchsize)))
    running_loss = 0.0

    for i in range (0,iteration):
        # get the inputs
        startIdx = i*batchsize
        endIdx = min((i+1)*batchsize, test_Num) # iterate should not exceed the test_Num
        inputs = testData[startIdx:endIdx,:]
        labels = testGT[startIdx:endIdx,:]

        inputs, labels = Variable(inputs.cuda().float()), Variable(labels.cuda().float())

        # forward
        output = net(inputs)

        loss = criterion(output, labels)
        running_loss += loss.data[0] 
        
        #sigmoid the output to make the output between [0,1].
        #The output is the result of the multi-label classification
        output=torch.nn.Sigmoid()(output)

        npoutput=output.data.cpu()
        npoutput=npoutput.numpy()
        outputs.extend(npoutput)
        
    #calculate the final loss
    final_loss = running_loss/test_Num*batchsize
    
    #calculate the precision recall and F1-score
    total_precision = 0
    total_recall = 0
    total_F1 = 0
    for i in range (0,test_Num):
        label = testGT[i].cpu().numpy()
        predict = outputs[i]
        #print(label)
        #print(predict)
        precision, recall, F1 = f1_score(label, predict)
        total_precision = total_precision+precision
        total_recall = total_recall+recall
        total_F1 = total_F1 + F1
    ave_precision = total_precision/test_Num
    ave_recall = total_recall/test_Num
    ave_F1 = total_F1/test_Num
    #print('final loss is %.6f' %(final_loss))
    return final_loss, ave_precision, ave_recall, ave_F1

def extract_feat(Data, keys, output_dir, num_attr=256):
    num = len(Data)
    if os.path.exists(os.path.join(output_dir, 'attributes.h5')):
        print('The file already exists in '+ output_dir)
        return 0

    with h5py.File(os.path.join(output_dir, 'attributes.h5')) as file_attr:

        for i in range(0, num):
            # get the inputs
            inputs = Data[i,:]
            inputs = Variable(inputs.cuda().float())

            # forward
            output = test_net(inputs)
            output = torch.nn.Sigmoid()(output)

            d_set_attr = file_attr.create_dataset(keys[i], (num_attr,), dtype="float") 
            d_set_attr[...] = output.data.cpu().float().numpy()

            if i % 500 == 0:
                print('processing %d/%d (%.2f%% done)' % (i, num, i*100.0 / num))
    
        file_attr.close()
    print('Finish extract features for '+output_dir)


################################################
#         define the input parameter           #
################################################
parser = argparse.ArgumentParser()
parser.add_argument('feat_files', nargs=2, type=str,
                    help='input files of features, first should be source, second is target.')
parser.add_argument('label_files', nargs=2, type=str,
                    help='input files of labels of attributes, source + target.')
parser.add_argument('--train', action='store_true', default=False)
args = parser.parse_args()

batchsize = 400
learning_rate = 0.03
epoch_number = 2000
GPU_ID ='0'
dir_attr = './attributes'
output_names = [x.split('/')[-4] for x in args.feat_files]
output_dirs = [os.path.dirname(x) for x in args.feat_files]

####################################################
#        read the COCO data and gt                 #
####################################################
#set the GPU ID
os.environ['CUDA_VISIBLE_DEVICES']=GPU_ID

#read the data and grount truth
feats = [h5py.File(x, 'r') for x in args.feat_files]
labels = [h5py.File(x, 'r') for x in args.label_files]

source_data = []
source_gt = []
target_data = []
target_gt = []

source_keys = feats[0].keys() # the cocoid's
shuffle(source_keys)
target_keys = feats[1].keys()

for key in source_keys:
    source_data.append(feats[0][key][:])
    source_gt.append(labels[0][key][:])

for key in target_keys:
    target_data.append(feats[1][key][:])
    target_gt.append(labels[1][key][:])    

source_data = np.array(source_data)
source_gt = np.array(source_gt)
target_data = np.array(target_data)
target_gt = np.array(target_gt)
print(source_data.shape)
print(target_data.shape)

####################################################
#     divide to training validation and testing    #
####################################################

#define the sample number
trainNum = source_data.shape[0]
testNum = target_data.shape[0]
classes = source_gt.shape[1]
print('Train number is '+str(trainNum))
print('Val number is '+str(testNum))
print('The number of classes is '+str(classes))
print('learning rate is '+str(learning_rate))
print('total number of epochs is '+str(epoch_number))

#prepare the train data
trainData = torch.from_numpy(source_data) # these are doubleTensors
trainGT = torch.from_numpy(source_gt) # also double, should be cast to float
# print(trainData)
# print(trainGT)
trainset = TensorDataset(trainData, trainGT)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

#prepare the test data
testData = torch.from_numpy(target_data)
testGT = torch.from_numpy(target_gt)

##########################################
#             Output features            #
##########################################

if not args.train:
    # define the network for extraction
    test_net = torch.load(dir_attr+'/models/'+output_names[0]+'_'+output_names[1]+'.model') # insert pretrained model
    test_net.cuda()
    test_net.eval()

    # for source
    extract_feat(trainData, source_keys, output_dir=output_dirs[0], num_attr=classes)
    extract_feat(testData, target_keys, output_dir=output_dirs[1], num_attr=classes)

    print('Finish Extracting features.')

else:
##########################################
#          training and testing          #
##########################################

    # define the network
    net = Net()
    net.cuda()
    net.train()

    #define the loss
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    print('Start Training')

    # initialization
    best_train_epoch = 0
    best_test_epoch = 0
    best_train_F1 = 0
    best_test_F1 = 0

    best_train_precision = 0
    best_test_precision = 0
    best_train_recall = 0
    best_test_recall = 0
    best_train_loss = 0
    best_test_loss = 0

    # start training 
    for epoch in range(epoch_number):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda().float()), Variable(labels.cuda().float())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        # ---------------------------------------
        # ----------- pseudo sampling -----------
        # ---------------------------------------




        # ---------------------------------------
        # ------------print out results ---------
        # ---------------------------------------

        print('new_epoch [%d,%d]  loss: %.6f' %(epoch + 1,epoch_number,  running_loss / (trainNum/batchsize)))
        running_loss = 0.0

        if epoch%10 == 9:
            train_loss, train_precision, train_recall, train_F1 = result(trainData,trainGT)
            print('epoch [%d,%d]  Training result: loss= %.6f, precision=%.5f, recall=%.5f, F1= %.5f' %(epoch + 1, epoch_number, train_loss, train_precision, train_recall, train_F1))
            test_loss, test_precision, test_recall, test_F1 = result(testData,testGT)
            print('epoch [%d,%d]  validation result: loss= %.6f, precision=%.5f, recall=%.5f, F1= %.5f'  %(epoch + 1, epoch_number, test_loss, test_precision, test_recall,test_F1))

            if train_F1 > best_train_F1:
                best_train_epoch = epoch + 1
                best_train_F1 = train_F1
                best_train_precision = train_precision
                best_train_recall = train_recall
                best_train_loss = train_loss
            if test_F1 > best_test_F1:
                best_test_epoch = epoch+1
                best_test_F1 = test_F1
                best_test_precision = test_precision
                best_test_recall = test_recall
                best_test_loss = test_loss
                torch.save(net, dir_attr+'/models/'+output_names[0]+'_'+output_names[1]+'_val.model')

                #torch.save(net, './model')
            print('Best training result(F1): epoch [%d,%d]  : loss= %.6f, precision=%.5f, recall=%.5f, F1= %.5f' %(best_train_epoch, epoch_number, best_train_loss, best_train_precision, best_train_recall, best_train_F1))
            print('Best validation result(F1): epoch [%d,%d]  : loss= %.6f, precision=%.5f, recall=%.5f, F1= %.5f' %(best_test_epoch, epoch_number, best_test_loss, best_test_precision, best_test_recall, best_test_F1))

    print('Finish Training')
    torch.save(net, dir_attr+'/models/'+output_names[0]+'_'+output_names[1]+'_nodann.model')
