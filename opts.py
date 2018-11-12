import argparse
import os

def parse_opt(domain='source'):
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('datasets', nargs=2, type=str,
                    help='input names of datasets, first should be source, second is target.')
    parser.add_argument('--input_opt', type=str, default='resnet101',
                    help='the input option of all mid-level features, source and target should be kept the same.')
    parser.add_argument('--data_path', type=str, default='../data',
                    help='the directory to store the original datasets.')
    parser.add_argument('--prepro_path', type=str, default='prepro',
                    help='the directory to store the correponding preprocess results.')
    parser.add_argument('--attribute_opt', type=str, default='',
                    help='to mark different attributes, used in debug.')

    # checkpoint
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)


    # Model settings
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--attribute_size', type=int, default=256,
                    help='the number of attributes')
    parser.add_argument('--use_fc', action='store_true', default=False,
                    help='the bool value indicating using or not using fc.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')


    # Optimization: General
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k, which includes the restval split')

    args = parser.parse_args()

    # define the input of source files or target files
    if not args.attribute_opt == '':
        args.attribute_opt = '_'+args.attribute_opt 
    if domain == 'target':
        args.input_target = args.datasets[1]
        args.input_target_opt = args.input_opt
        args.input_json = os.path.join(args.data_path, args.input_target, args.prepro_path, args.input_target_opt, args.input_target+'.json')
        args.input_label_h5 = os.path.join(args.data_path, args.input_target, args.prepro_path, args.input_target_opt, args.input_target+'.h5')
        args.input_att_dir = os.path.join(args.data_path, args.input_target, args.prepro_path, args.input_target_opt)
        args.input_fc_dir = os.path.join(args.data_path, args.input_target, args.prepro_path, args.input_target_opt)
        # args.input_attributes_dir = os.path.join('./attributes/', args.input_target)
        args.input_attributes_dir = os.path.join(args.data_path, args.input_target, args.prepro_path, args.input_target_opt)
    else:
        args.input_source = args.datasets[0]
        args.input_source_opt = args.input_opt
        args.input_json = os.path.join(args.data_path, args.input_source, args.prepro_path, args.input_source_opt, args.input_source+'.json')
        args.input_label_h5 = os.path.join(args.data_path, args.input_source, args.prepro_path, args.input_source_opt, args.input_source+'.h5')
        args.input_att_dir = os.path.join(args.data_path, args.input_source, args.prepro_path, args.input_source_opt)
        args.input_fc_dir = os.path.join(args.data_path, args.input_source, args.prepro_path, args.input_source_opt)
        # args.input_attributes_dir = os.path.join('./attributes/', args.input_source)
        args.input_attributes_dir = os.path.join(args.data_path, args.input_source, args.prepro_path, args.input_source_opt)

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args