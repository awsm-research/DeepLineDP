import os, sys, argparse, re

import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from tqdm import tqdm

# for importing file from previous directory
sys.path.append('../')

from my_util import *

arg = argparse.ArgumentParser()
arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-epochs', type=int, default=100)
arg.add_argument('-target_epochs', type=str, default=100, help='which epoch of model to load')
arg.add_argument('-train',action='store_true')
arg.add_argument('-predict',action='store_true')

args = arg.parse_args()

torch.manual_seed(0)

# model parameters

batch_size = 8
output_size = 1
embedding_dim = 50
hidden_dim = 256
lr = 0.001
epochs = args.epochs

criterion = nn.BCELoss()


save_every_epochs = 1 # default is 5

max_seq_len = 50

save_model_dir = '../../output/model/Bi-LSTM/'
save_prediction_dir = '../../output/prediction/Bi-LSTM/'
loss_dir = '../../output/loss/Bi-LSTM/'

if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)

if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch
        output_size : 1
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension word embeddings
        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        # Initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.lstm = nn.LSTM(embedding_length, hidden_size,bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, input_tensor):

        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = self.word_embeddings(input_tensor.type(torch.LongTensor).cuda()) 

        # input.size() = (num_sequences, batch_size, embedding_length)
        input = input.permute(1, 0, 2) 
        h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda()) # Initialize hidden state of the LSTM
        c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda()) # Initialize cell state of the LSTM

        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        output = self.fc(final_hidden_state[-1]) # the last hidden state is output of lstm model
        
        sig_out = self.sig(output)
        
        return sig_out

def pad_features(codevec, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## getting the correct rows x cols shape
    features = np.zeros((len(codevec), seq_length), dtype=int)
    
    ## for each review, I grab that review
    for i, row in enumerate(codevec):
        if len(row) > seq_length:
            features[i,:] = row[:seq_length]
        else:
            features[i, :] = row + [0]* (seq_length - len(row))
    
    return features

def get_dataloader_for_LSTM(w2v_model, code,encoded_labels):
    codevec = []
    for c in code:
        codevec.append([w2v_model.wv.vocab[word].index if word in w2v_model.wv.vocab else len(w2v_model.wv.vocab) for word in c.split()])

    max_seq_len = max([len(cv) for cv in codevec])

    # to prevent out of memory error
    if max_seq_len > 40000:
        max_seq_len = 40000
        
    features = pad_features(codevec, seq_length=max_seq_len)
    tensor_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(encoded_labels))
    dl = DataLoader(tensor_data, shuffle=True, batch_size=batch_size,drop_last=True)
    return dl

def train_model(dataset_name):

    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    train_release = all_train_releases[dataset_name]
    valid_release = all_eval_releases[dataset_name][0]
    
    train_df = get_df_for_baseline(train_release)
    train_code, train_encoded_labels = get_data_and_label(train_df)

    valid_df = get_df_for_baseline(valid_release)
    valid_code, valid_encoded_labels = get_data_and_label(valid_df)

    word2vec_file_dir = os.path.join('.'+word2vec_baseline_file_dir,dataset_name+'.bin')

    word2vec_model = Word2Vec.load(word2vec_file_dir)
    
    vocab_size = len(word2vec_model.wv.vocab)+1
        
    train_dl = get_dataloader_for_LSTM(word2vec_model, train_code,train_encoded_labels)
    valid_dl = get_dataloader_for_LSTM(word2vec_model, valid_code,valid_encoded_labels)

    net = LSTMClassifier(batch_size, output_size, hidden_dim, vocab_size, embedding_dim)

    net = net.cuda()
    
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = nn.BCELoss()

    checkpoint_files = os.listdir(actual_save_model_dir)

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')

    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:
        word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec_model, embedding_dim)
        net.word_embeddings.weight = nn.Parameter(word2vec_weights)

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
    # for e in tqdm(range(3)): # for testing (will remove later...)
        # batch loop

        train_losses = []
        val_losses = []

        net.train()

        for inputs, labels in train_dl:
            
            inputs, labels = inputs.cuda(), labels.cuda()

            net.zero_grad()
            
            output = net(inputs)
            
            # calculate the loss and perform backprop
            loss = criterion(output, labels.reshape(-1,1).float())

            train_losses.append(loss.item())

            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

        train_loss_all_epochs.append(np.mean(train_losses))

        with torch.no_grad():

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
    
    loss_df.to_csv(loss_dir+dataset_name+'-Bi-LSTM-loss_record.csv',index=False)

    print('finished training model of',dataset_name)

    
# epoch (int): which epoch to load model
def predict_defective_files_in_releases(dataset_name, target_epochs = 100):

    actual_save_model_dir = save_model_dir+dataset_name+'/'

    train_rel = all_train_releases[dataset_name]
    eval_rel = all_eval_releases[dataset_name][1:]

    word2vec_file_dir = os.path.join('.'+word2vec_baseline_file_dir,dataset_name+'.bin')

    word2vec_model = Word2Vec.load(word2vec_file_dir)
    
    vocab_size = len(word2vec_model.wv.vocab) + 1

    net = LSTMClassifier(batch_size, output_size, hidden_dim, vocab_size, embedding_dim)

    checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+target_epochs+'epochs.pth')

    net.load_state_dict(checkpoint['model_state_dict'])

    net = net.cuda()
    
    net.eval()
    
    for rel in eval_rel:

        test_df = get_df_for_baseline(rel)
        code,encoded_labels = get_data_and_label(test_df)
        test_dl = get_dataloader_for_LSTM(word2vec_model, code,encoded_labels)

        y_pred = []
        y_test = []
        y_prob = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_dl):
                inputs, labels = inputs.cuda(), labels.cuda()
                output = net(inputs)

                pred = torch.round(output)
#                     print('pred',pred)

                y_pred.extend(pred.cpu().numpy().squeeze().tolist())
                y_test.extend(labels.cpu().numpy().squeeze().tolist())
                y_prob.extend(output.cpu().numpy().squeeze().tolist())

        prediction_df = pd.DataFrame()
        prediction_df['train_release'] = [train_rel]*len(y_test)
        prediction_df['test_release'] = [rel]*len(y_test)
        prediction_df['actual'] = y_test
        prediction_df['pred'] = y_pred
        prediction_df['prob'] = y_prob
        
        print('finished release',rel)

        prediction_df.to_csv(save_prediction_dir+rel+'.csv',index=False)

    print('-'*100,'\n')

proj_name = args.dataset

if args.train:
    train_model(proj_name)

if args.predict:
    target_epochs = args.target_epochs
    predict_defective_files_in_releases(proj_name, target_epochs)
