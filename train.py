from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    opt['source'].use_att = utils.if_use_att(opt['source'].caption_model)
    loader = {}
    loader['source'] = DataLoader(opt['source'])
    opt['source'].vocab_size = loader['source'].vocab_size # vocab and seq_length all follows the source dataset
    opt['source'].seq_length = loader['source'].seq_length
    loader['target'] = DataLoader(opt['target'], target=True, opt_extra=opt['source'])

    tf_summary_writer = tf and tf.summary.FileWriter(opt['source'].checkpoint_path)

    infos = {}
    histories = {}
    if opt['source'].start_from is not None: # the state of previous training session
        # open old infos and check if models are compatible
        with open(os.path.join(opt['source'].start_from, 'infos_'+opt['source'].id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']['source']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same: # the key arguments of RNN model should keep the same
                assert vars(saved_model_opt)[checkme] == vars(opt['source'])[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt['source'].start_from, 'histories_'+opt['source'].id+'.pkl')):
            with open(os.path.join(opt['source'].start_from, 'histories_'+opt['source'].id+'.pkl')) as f:
                histories = cPickle.load(f) # it is the histories file that be used in the following training

    iteration = infos.get('iter', 0) # obtain the iteration number, if not defined, then return 0
    epoch = infos.get('epoch', 0) # obtain the epoch number, if not defined, then return 0

    val_result_history = histories.get('val_result_history', {}) # if not defined, return {}
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader['source'].iterator = infos.get('iterator', loader['source'].iterator) # if not defined, return loader.iterator
    loader['source'].split_ix = infos.get('split_ix', loader['source'].split_ix)
    if opt['source'].load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt['source']) # opt contains the model to use, here is the caption training model, the definitions are in __init__.py
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion() # for later learning

    optimizer = optim.Adam(model.parameters(), lr=opt['source'].learning_rate, weight_decay=opt['source'].weight_decay)

    # Load the optimizer, if training from another status instead of scratch
    if vars(opt['source']).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt['source'].start_from, 'optimizer.pth')))

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt['source'].learning_rate_decay_start and opt['source'].learning_rate_decay_start >= 0:
                frac = (epoch - opt['source'].learning_rate_decay_start) // opt['source'].learning_rate_decay_every
                decay_factor = opt['source'].learning_rate_decay_rate  ** frac
                opt['source'].current_lr = opt['source'].learning_rate * decay_factor
                utils.set_lr(optimizer, opt['source'].current_lr) # set the decayed rate
            else:
                opt['source'].current_lr = opt['source'].learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt['source'].scheduled_sampling_start and opt['source'].scheduled_sampling_start >= 0:
                frac = (epoch - opt['source'].scheduled_sampling_start) // opt['source'].scheduled_sampling_increase_every
                opt['source'].ss_prob = min(opt['source'].scheduled_sampling_increase_prob  * frac, opt['source'].scheduled_sampling_max_prob)
                model.ss_prob = opt['source'].ss_prob
            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader['source'].get_batch()
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp
        
        optimizer.zero_grad()
        loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
        loss.backward()
        utils.clip_gradient(optimizer, opt['source'].grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt['source'].losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt['source'].current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt['source'].current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt['source'].save_checkpoint_every == 0): 
            # eval model
            eval_kwargs = {'dataset': opt['target'].input_target}
            eval_kwargs.update(vars(opt['target'])) # the final version of eval_kwargs is a dict same as opt added with the previous key
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader['target'], eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k,v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt['target'].language_eval == 1:
                current_score = lang_stats['CIDEr'] # only based on CIDEr.
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score: # choose the best model based on the validation score
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt['source'].checkpoint_path, 'model'+str(iteration)+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt['source'].checkpoint_path, 'optimizer'+str(iteration)+'.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterator'] = loader['source'].iterator
                infos['split_ix'] = loader['source'].split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader['source'].get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt['source'].checkpoint_path, 'infos_'+opt['source'].id+str(iteration)+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt['source'].checkpoint_path, 'histories_'+opt['source'].id+str(iteration)+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt['source'].checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt['source'].checkpoint_path, 'infos_'+opt['source'].id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt['source'].max_epochs and opt['source'].max_epochs != -1:
            break

opt = {}
opt['source'] = opts.parse_opt()
opt['target'] = opts.parse_opt('target')

train(opt)
