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

# dir_suffix = 'no-abs-rebalancing-adaptive-ratio2-with-comment'
dir_suffix = 'rebalancing-adaptive-ratio2'

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

# def prepare_data(rel):
#     df = pd.read_csv(file_lvl_gt+rel+'.txt',sep='\t')
#     df = df.dropna()
    
#     code = list(df['Code'])
    
#     code_3D_list = create3DList(code)
#     label = list(df['Bug'])
    
#     return code_3D_list, label

def get_combined_code(rel, include_comment=False, include_blank_line=False):
    df = pd.read_csv(file_lvl_gt+rel+".csv")
    print(df.head())

    if not include_comment:
        df = df[df['is_comment']==False]

    if not include_blank_line:
        df = df[df['is_blank']==False]

    df = df.fillna('')
    
    for filename, group_df in df.groupby('filename'):
        print(filename)
        print(group_df)

        code = list(group_df['code_line'])
        print(code)
        break

def train_model(dataset_name):

    dataset_name = 'activemq'
    loss_dir = '../output/loss/'+dir_suffix+'/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    get_combined_code(train_rel)