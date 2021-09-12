import os, re, argparse

import torch.optim as optim
# from torch.utils.data import WeightedRandomSampler

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

# from sklearn.utils import compute_class_weight

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
# dir_suffix = 'rebalancing-adaptive-ratio2' # wrong one...
dir_suffix = 'train-test-subsequent-release'

if include_comment:
    dir_suffix = dir_suffix + '-with-comment'

if include_blank_line:
    dir_suffix = dir_suffix + '-with-blank-line'

if include_test_file:
    dir_suffix = dir_suffix + '-with-test-file'

dir_suffix = dir_suffix+'-'+str(embed_dim)+'-dim'

prediction_dir = '../output/prediction/DeepLineDP/'+dir_suffix+'/'
save_model_dir = '../output/model/DeepLineDP/'+dir_suffix+'/'

file_lvl_gt = '../datasets/preprocessed_data/'


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
    
    valid_df = get_df(valid_rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line)

    train_code3d, train_label = get_code3d_and_label(train_df)
    valid_code3d, valid_label = get_code3d_and_label(valid_df)

    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    vocab_size = len(word2vec.wv.vocab)  + 1 # for unknown tokens

    x_train_vec = get_x_vec(train_code3d, word2vec)
    x_valid_vec = get_x_vec(valid_code3d, word2vec)

    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)

    train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len)
 
    valid_dl = get_dataloader(x_valid_vec, valid_label,batch_size,max_sent_len)

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

    model = model.cuda()
    model.sent_attention.word_attention.freeze_embeddings(False)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion = nn.BCELoss()

    checkpoint_files = os.listdir(actual_save_model_dir)

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')

    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:
        model.sent_attention.word_attention.init_embeddings(word2vec_weights)
        current_checkpoint_num = 1

        train_loss_all_epochs = []
        val_loss_all_epochs = []
    
    else:
        checkpoint_nums = [int(re.findall('\d+',s)[0]) for s in checkpoint_files]
        current_checkpoint_num = max(checkpoint_nums)

        checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+str(current_checkpoint_num)+'epochs.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        loss_df = pd.read_csv(loss_dir+dataset_name+'-loss_record.csv')
        train_loss_all_epochs = list(loss_df['train_loss'])
        val_loss_all_epochs = list(loss_df['valid_loss'])

        current_checkpoint_num = current_checkpoint_num+1 # go to next epoch
        print('train model from epoch',current_checkpoint_num)

    # for each epoch

    for epoch in tqdm(range(current_checkpoint_num,num_epochs+1)):
        train_losses = []
        val_losses = []

        model.train()

        for inputs, labels in train_dl:

            inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()
            output, _, __ = model(inputs_cuda)

            # weight_tensor = get_loss_weight(labels)

            # criterion.weight = weight_tensor

            loss = criterion(output, labels_cuda.reshape(batch_size,1))

            train_losses.append(loss.item())
            
            torch.cuda.empty_cache()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            torch.cuda.empty_cache()

        train_loss_all_epochs.append(np.mean(train_losses))

        with torch.no_grad():
            criterion.weight = None
            
            for inputs, labels in valid_dl:

                inputs, labels = inputs.cuda(), labels.cuda()
                output, _, __ = model(inputs)
            
                val_loss = criterion(output, labels.reshape(batch_size,1))

                val_losses.append(val_loss.item())

            val_loss_all_epochs.append(np.mean(val_losses))

        if epoch % save_every_epochs == 0:
            print(dataset_name,'- at epoch:',str(epoch))

            if exp_name == '':
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+str(epoch)+'epochs.pth')
            else:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+exp_name+'_'+str(epoch)+'epochs.pth')

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(loss_dir+dataset_name+'-loss_record.csv',index=False)

dataset_name = args.dataset
train_model(dataset_name)