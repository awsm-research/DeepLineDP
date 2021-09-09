from DeepLineDP.script.train_word2vec import get_w2v_path
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

weight_dict = {}

# model setting
batch_size = 32
num_epochs = 100
max_grad_norm = 5
embed_dim = 50
word_gru_hidden_dim = 64
sent_gru_hidden_dim = 64
word_gru_num_layers = 1
sent_gru_num_layers = 1
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = 0.5
lr = 0.001

save_every_epochs = 5
exp_name = ''

max_train_LOC = 900

include_comment = True
include_blank_line = False
include_test_file = False

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
    
# labels is a tensor of label
def get_loss_weight(labels):
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1,1).cuda()
    return weight_tensor


def train_model(dataset_name):

    # dataset_name = 'activemq'
    loss_dir = '../output/loss/'+dir_suffix+'/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    w2v_dir = get_w2v_path(include_comment=include_comment,include_test_file=include_test_file)

    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    train_df = get_df(train_rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line)
    valid_df = get_df(train_rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line)

    train_code3d, train_label = get_code3d_and_label(train_df)
    valid_code3d, valid_label = get_code3d_and_label(valid_df)

    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_label), y = train_label)

    weight_dict['defect'] = sample_weights[1]
    weight_dict['clean'] = sample_weights[0]


    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    vocab_size = len(word2vec.wv.vocab)  + 1 # for unknown tokens