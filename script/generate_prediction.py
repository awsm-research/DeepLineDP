import os, re, argparse

import torch.optim as optim

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

from sklearn.utils import compute_class_weight

from DeepLineDP_model import *
from my_util import *

torch.manual_seed(0)

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-batch_size', type=int, default=32)
arg.add_argument('-num_epochs', type=int, default=100)
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size')
arg.add_argument('-word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
arg.add_argument('-sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
arg.add_argument('-word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
arg.add_argument('-sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
arg.add_argument('-dropout', type=float, default=0.5, help='dropout rate')
arg.add_argument('-lr', type=float, default=0.001, help='learning rate')
arg.add_argument('-exp_name',type=str,default='')

arg.add_argument('-include_comment',action='store_true')
arg.add_argument('-include_blank_line',action='store_true')
arg.add_argument('-include_test_file',action='store_true')

args = arg.parse_args()

weight_dict = {}

# model setting
batch_size = args.batch_size
num_epochs = args.num_epochs
max_grad_norm = 5
embed_dim = args.embed_dim
word_gru_hidden_dim = args.word_gru_hidden_dim
sent_gru_hidden_dim = args.sent_gru_hidden_dim
word_gru_num_layers = args.word_gru_num_layers
sent_gru_num_layers = args.sent_gru_num_layers
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = args.dropout
lr = args.lr

save_every_epochs = 5
exp_name = args.exp_name

max_train_LOC = 900

include_comment = args.include_comment
include_blank_line = args.include_blank_line
include_test_file = args.include_test_file

# dir_suffix = 'no-abs-rebalancing-adaptive-ratio2-with-comment'
dir_suffix = 'rebalancing-adaptive-ratio2'

if include_comment:
    dir_suffix = dir_suffix + '-with-comment'

if include_blank_line:
    dir_suffix = dir_suffix + '-with-blank-line'

if include_test_file:
    dir_suffix = dir_suffix + '-with-test-file'

prediction_dir = '../output/prediction/DeepLineDP/'+dir_suffix+'/'
save_model_dir = '../output/model/DeepLineDP/'+dir_suffix+'/'

file_lvl_gt = '../datasets/preprocessed_data/'


if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

def predict_defective_files_in_releases(dataset_name, target_epochs):

    batch_size = 8

    actual_save_model_dir = save_model_dir+dataset_name+'/'

    train_rel = all_train_releases[dataset_name]
    test_rel = all_eval_releases[dataset_name][1:]
    
    test_code_3D_list_dict = {}
    x_test_vec_dict = {}
    test_label_dict = {}
    
    all_x_vec = []
    
    w2v_dir = get_w2v_path(include_comment=include_comment,include_test_file=include_test_file)

    word2vec = Word2Vec.load(w2v_dir)

    print('load Word2Vec for',dataset_name,'finished')

    total_vocab = len(word2vec.wv.vocab)

    vocab_size = total_vocab +1 # for unknown tokens

        
    max_sent_len = 999999
    
        
    model = HierarchicalAttentionNetwork(
        num_classes=1,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=word_gru_num_layers,
        sent_gru_num_layers=sent_gru_num_layers,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout)

    if exp_name == '':
        checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+target_epochs+'epochs.pth')

    else:
        checkpoint = torch.load(actual_save_model_dir+exp_name+'/checkpoint_'+target_epochs+'epochs.pth')

    model.load_state_dict(checkpoint['model_state_dict'])

    
    model.sent_attention.word_attention.freeze_embeddings(True)

    model = model.cuda()
    model.eval()

    # for rel in test_rel:
        # print('evaluating release:', rel)
        
        # test_dl = get_dataloader(x_test_vec_dict[rel], test_label_dict[rel], batch_size,max_sent_len)

        # y_pred = []
        # y_test = []
        # y_prob = []

        # with torch.no_grad():
        #     for inputs, labels in tqdm(test_dl):
        #         inputs, labels = inputs.cuda(), labels.cuda()
        #         output, word_att_weights, sent_att_weights = model(inputs)

        #         pred = torch.round(output)

        #         y_pred.extend(pred.cpu().numpy().squeeze().tolist())
        #         y_test.extend(labels.cpu().numpy().squeeze().tolist())
        #         y_prob.extend(output.cpu().numpy().squeeze().tolist())

        #         torch.cuda.empty_cache()

        # prediction_df = pd.DataFrame()
        # prediction_df['train_release'] = [train_rel]*len(y_test)
        # prediction_df['test_release'] = [rel]*len(y_test)
        # prediction_df['actual'] = y_test
        # prediction_df['pred'] = y_pred
        # prediction_df['prob'] = y_prob

        # if exp_name == '':
        #     prediction_df.to_csv(prediction_dir+rel+'_'+target_epochs+'_epochs.csv',index=False)
        # else:
        #     prediction_df.to_csv(prediction_dir+rel+'_'+exp_name+'_'+target_epochs+'_epochs.csv',index=False)

        # print('finished predicting defective files in',rel)