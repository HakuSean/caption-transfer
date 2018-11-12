# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel

class OldModel(CaptionModel):
    def __init__(self, opt):
        super(OldModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.attribute_size = opt.attribute_size
        self.att_feat_size = opt.att_feat_size
        self.use_fc = opt.use_fc

        self.ss_prob = 0.0 # Schedule sampling probability

        if self.use_fc:
            self.linear = nn.Linear(self.fc_feat_size + self.attribute_size, self.num_layers * self.rnn_size) # feature to rnn_size
        else:
            self.linear = nn.Linear(self.attribute_size, self.num_layers * self.rnn_size) # feature to rnn_size            
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1) # transfer the output of rnn to vocab, output: batch_size * vocab_size+1
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0) # if batch_size is 1, output weights, but batch_size cannot be 1, always 5.
        state = self.init_hidden(fc_feats)

        outputs = []
        weights = []

        for i in range(seq.size(1) - 1): # length of a sentence
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()    # in order not to change the original seq      
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it) # batch_size * vocab+1 -> batch_size * input_encoding_size

            output, state, weight = self.core(xt, fc_feats, att_feats, state) # state will become the input next time
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output) # outputs should have seq_length number of output, each output should be batch_size * vocab_size+1
            weights.append(weight)

        if batch_size == 1:
            return torch.cat([_.unsqueeze(1) for _ in weights], 1) # batch_size * seq_length * att_size(196=14*14)
        else:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1) # batch_size * seq_length * vocab_size +1 

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state, weight = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        if self.use_fc:
            feat_size = self.fc_feat_size+self.attribute_size
        else:
            feat_size = self.attribute_size

        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, feat_size) # from 1*feat_size to n * feat_size
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            
            state = self.init_hidden(tmp_fc_feats)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            done_beams = []
            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state, weight = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_() # start of sentence, batch_size zeros.
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1) # find the largest probability for each sentence, sampleLogprobs is the value, it is the index.
                it = it.view(-1).long() # view: remove the -1 dim. it: (batch_size,1)-> (batch_size, )
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False)) # input of the next step

            # If it is not 0, then continue
            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0: # if all are finished, then stop; else, continue. This means in a batch, the output will have the same length.
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state, weight = self.core(xt, fc_feats, att_feats, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output))) # for each batch, each word is assigned a different probability, the logprobs is batch_size * (vocab+1)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1) # seq: batch * length(any length), seqLogprobs: same size 


class ShowAttendTellCore(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.attribute_size = opt.attribute_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.use_fc = opt.use_fc
        
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)  

        if self.att_hid_size > 0:
            if self.use_fc:
                self.attr2att = nn.Linear(self.attribute_size, self.att_hid_size)
            # else:
            #     self.attr2att = nn.Linear(self.fc_feat_size, self.att_hid_size)

            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1) # only one output
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    def forward(self, xt, fc_feats, att_feats, state):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size # batch*14*14*2048//batch//2048 = the size of map but flatten=196
        att = att_feats.view(-1, self.att_feat_size)            # (batch * att_size) * 2048
        if self.use_fc:
            attr = fc_feats[:, self.fc_feat_size:]
        if self.att_hid_size > 0:
            att = self.ctx2att(att)                             # (batch * att_size) * att_hid_size
            att = att.view(-1, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
            att_h = self.h2att(state[0][-1])                    # batch * att_hid_size, input is the 
            att_h = att_h.unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
            if self.use_fc:
                att_attr = self.attr2att(attr)
                att_attr = att_attr.unsqueeze(1).expand_as(att)
                dot = att + att_h + att_attr
            else:
                dot = att + att_h                                   # batch * att_size * att_hid_size
            dot = F.tanh(dot)                                   # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
            dot = self.alpha_net(dot)                           # (batch * att_size) * 1
            dot = dot.view(-1, att_size)                        # batch * att_size
        else:
            att = self.ctx2att(att)(att)                        # (batch * att_size) * 1
            att = att.view(-1, att_size)                        # batch * att_size
            att_h = self.h2att(state[0][-1])                    # batch * 1
            att_h = att_h.expand_as(att)                        # batch * att_size
            dot = att_h + att                                   # batch * att_size
        
        weight = F.softmax(dot) # batch * att_size, after unsqueeze: batch * 1 * att_size
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state) # xt = batch * input_coding_size, so the input is 1*batch*(att_feat_size+input_coding_size). Since there is no batch_first parameter, then the sequence is 1, and batch = batch_size.
        return output.squeeze(0), state, weight.squeeze(1)

class AllImgCore(nn.Module):
    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.attribute_size = opt.attribute_size
        self.use_fc = opt.use_fc
        
        if self.use_fc:
            self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size + self.attribute_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        else:
            self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)            

    def forward(self, xt, fc_feats, att_feats, state):
        output, state = self.rnn(torch.cat([xt, fc_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state

class ShowAttendTellModel(OldModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)

class AllImgModel(OldModel):
    def __init__(self, opt):
        super(AllImgModel, self).__init__(opt)
        self.core = AllImgCore(opt)

