import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import functional as F

import pandas as pd
import os, sys ,argparse, re

from gensim.models import Word2Vec

from tqdm import tqdm

from baseline_util import *

sys.path.append('../')

from my_util import *

arg = argparse.ArgumentParser()
arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-epochs', type=int, default=30)
arg.add_argument('-target_epochs', type=str, default='6', help='which epoch of model to load')
arg.add_argument('-exp_name',type=str,default='')
arg.add_argument('-train',action='store_true')
arg.add_argument('-predict',action='store_true')

args = arg.parse_args()


# model parameters

batch_size = 32
embed_dim = 50
n_filters = 100      # default is 100
lr = 0.001

epochs = args.epochs 
save_every_epochs = 1
max_seq_len = 50 # number of tokens of each line in a file
max_train_LOC = 900 # max length of all code in the whole dataset

exp_name = args.exp_name

save_model_dir = '../../output/model/CNN/'
save_prediction_dir = '../../output/prediction/CNN/'
    
if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)

class CNN(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels,  keep_probab, vocab_size, embedding_dim):
        super(CNN, self).__init__()
        '''
        Arguments
        ---------
        batch_size : Size of each batch
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        '''
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vocab_size = vocab_size
        self.embedding_length = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.conv = nn.Conv2d(1, 100, (5, embedding_dim))


        self.dropout = nn.Dropout(keep_probab)
        self.fc = nn.Linear(100, 1)

        self.sig = nn.Sigmoid()

    def conv_block(self, input, conv_layer):

        conv_out = F.relu(conv_layer(input))
        conv_out = torch.squeeze(conv_out,-1)
        max_out = F.max_pool1d(conv_out, conv_out.size()[2])

        return max_out

    def forward(self, input_tensor):
        '''
        Parameters
        ----------
        input_tensor: input_tensor of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        '''
        # input.size() = (batch_size, num_seq, embedding_length)
        input = self.word_embeddings(input_tensor.type(torch.LongTensor).cuda())
        
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        input = input.unsqueeze(1)
        

        max_out = self.conv_block(input, self.conv)
        max_out = max_out.view(max_out.size(0),-1)

        fc_in = self.dropout(max_out)

        logits = self.fc(fc_in)
        sig_out = self.sig(logits)

        return sig_out
                                            

def train_model(dataset_name):

    loss_dir = '../../output/loss/CNN/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    w2v_dir = get_w2v_path()

    w2v_dir = os.path.join('../'+w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    train_df = get_df(train_rel,is_baseline=True)
    
    valid_df = get_df(valid_rel,is_baseline=True)

    word2vec_model = Word2Vec.load(w2v_dir)
    
    vocab_size = len(word2vec_model.wv.vocab)  + 1 # for unknown tokens

    train_code, train_label = prepare_data(train_df, to_lowercase = True)
    valid_code, valid_label = prepare_data(valid_df, to_lowercase = True)

    word2vec_model = Word2Vec.load(w2v_dir)

    padding_idx = word2vec_model.wv.vocab['<pad>'].index

    vocab_size = len(word2vec_model.wv.vocab)+1
        
    train_dl = get_dataloader(word2vec_model, train_code,train_label, padding_idx)
    valid_dl = get_dataloader(word2vec_model, valid_code,valid_label, padding_idx)

    net = CNN(batch_size, 1, n_filters, 0.5, vocab_size, embed_dim)

    net = net.cuda()
    
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = nn.BCELoss()

    checkpoint_files = os.listdir(actual_save_model_dir)

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')

    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:

        current_checkpoint_num = 1

        train_loss_all_epochs = []
        val_loss_all_epochs = []

    
    else:
        checkpoint_nums = [int(re.findall('\d+',s)[0]) for s in checkpoint_files]
        current_checkpoint_num = max(checkpoint_nums)

        checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+str(current_checkpoint_num)+'epochs.pth')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        loss_df = pd.read_csv(loss_dir+dataset_name+'-Bi-LSTM-loss_record.csv')
        train_loss_all_epochs = list(loss_df['train_loss'])
        val_loss_all_epochs = list(loss_df['valid_loss'])

        current_checkpoint_num = current_checkpoint_num+1 # go to next epoch

        print('train model from epoch',current_checkpoint_num)
    
    clip=5 # gradient clipping

    print('training model of',dataset_name)

    for e in tqdm(range(current_checkpoint_num,epochs+1)):
        train_losses = []
        val_losses = []

        net.train()

        # batch loop
        for inputs, labels in train_dl:

            inputs, labels = inputs.cuda(), labels.cuda()
            net.zero_grad()
            
            # get the output from the model
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output, labels.reshape(-1,1).float())
            train_losses.append(loss.item())

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            

        train_loss_all_epochs.append(np.mean(train_losses))

        with torch.no_grad():

            net.eval()

            for inputs, labels in valid_dl:

                inputs, labels = inputs.cuda(), labels.cuda()
                output = net(inputs)

                val_loss = criterion(output, labels.reshape(batch_size,1).float())

                val_losses.append(val_loss.item())

            val_loss_all_epochs.append(np.mean(val_losses))

        if e % save_every_epochs == 0: 
            torch.save({
                        'epoch': e,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, 
                        actual_save_model_dir+'checkpoint_'+str(e)+'epochs.pth')
    

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(loss_dir+dataset_name+'-CNN-loss_record.csv',index=False)

    print('finished training model of',dataset_name)


# target_epochs (int): which epoch to load model
def predict_defective_files_in_releases(dataset_name, target_epochs = 100):
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    w2v_dir = get_w2v_path()

    w2v_dir = os.path.join('../'+w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    train_rel = all_train_releases[dataset_name]
    eval_rels = all_eval_releases[dataset_name][1:]

    word2vec_model = Word2Vec.load(w2v_dir)
    
    vocab_size = len(word2vec_model.wv.vocab) + 1

    net = CNN(batch_size, 1, n_filters, 0.5, vocab_size, embed_dim)

    checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+target_epochs+'epochs.pth')

    net.load_state_dict(checkpoint['model_state_dict'])

    net = net.cuda()
    
    net.eval()
    
    for rel in eval_rels:
        row_list = []

        test_df = get_df(rel, is_baseline=True)

        for filename, df in tqdm(test_df.groupby('filename')):

            file_label = bool(df['file-label'].unique())

            code = list(df['code_line'])

            code_str = get_code_str(code, True)
            code_list = [code_str]

            code_vec = get_code_vec(code_list, word2vec_model)

            code_tensor = torch.tensor(code_vec)

            output = net(code_tensor)
            file_prob = output.item()
            prediction = bool(round(file_prob))

            row_dict = {
                        'project': dataset_name, 
                        'train': train_rel, 
                        'test': rel, 
                        'filename': filename, 
                        'file-level-ground-truth': file_label, 
                        'prediction-prob': file_prob, 
                        'prediction-label': prediction
                        }

            row_list.append(row_dict)

        df = pd.DataFrame(row_list)
        df.to_csv(save_prediction_dir+rel+'-'+target_epochs+'-epochs.csv', index=False)

proj_name = args.dataset
if args.train:
    train_model(proj_name)

if args.predict:
    target_epochs = args.target_epochs
    predict_defective_files_in_releases(proj_name, target_epochs)
