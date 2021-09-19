import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

import pandas as pd
import os, sys ,argparse, re

from gensim.models import Word2Vec

from tqdm import tqdm

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
output_size = 1
embed_dim = 50
n_filters = 128      # default is 128
kernel_size = [5,5,5] # for 3 conv layers
lr = 0.001

epochs = args.epochs # default is 100
save_every_epochs = 2 # default is 5
max_seq_len = 50 # number of tokens of each line in a file
max_train_LOC = 900 # max length of all code in the whole dataset

exp_name = args.exp_name

save_every_epochs = 2 # default is 5


include_comment = True
include_blank_line = False
include_test_file = False

to_lowercase = True

dir_suffix = 'lowercase'

if include_comment:
    dir_suffix = dir_suffix + '-with-comment'

if include_blank_line:
    dir_suffix = dir_suffix + '-with-blank-line'

if include_test_file:
    dir_suffix = dir_suffix + '-with-test-file'

dir_suffix = dir_suffix+'-'+str(embed_dim)+'-dim'


save_model_dir = '../../output/model/CNN/'
save_prediction_dir = '../../output/prediction/CNN/'
    
if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)

class CNN(nn.Module):
    def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, keep_probab, vocab_size, embedding_dim):
        super(CNN, self).__init__()
        '''
        Arguments
        ---------
        batch_size : Size of each batch
        output_size : 1
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 2 different kernel_heights. Convolution will be performed 2 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        '''
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.vocab_size = vocab_size
        self.embedding_length = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_dim))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_dim))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_dim))

        self.dropout = nn.Dropout(keep_probab)
        self.fc = nn.Linear(len(kernel_heights)*out_channels, output_size)
        self.sig = nn.Sigmoid()

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input) # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3)) # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input_tensor):
        '''
        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        '''

        input = self.word_embeddings(input_tensor.type(torch.LongTensor).cuda())
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        # all_out = torch.cat((max_out1, max_out2), 1)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.fc(fc_in)
        sig_out = self.sig(logits)
        return sig_out
                                            

# def pad_code(code_list_3d,max_sent_len):
#     paded = []
    
#     for file in code_list_3d:
#         sent_list = []
#         for line in file:
#             new_line = line
#             if len(line) > max_seq_len:
#                 new_line = line[:max_seq_len]
#             else:
#                 new_line = line+[0]*(max_seq_len - len(line))
#             sent_list.extend(new_line)
            
#         if max_sent_len-len(file) > 0:
#             for i in range(0,max_sent_len-len(file)):
#                 sent_list.extend([0]*max_seq_len)
                
#         paded.append(sent_list[:max_sent_len])

#     return paded


# def get_dataloader_for_CNN(code3d, encoded_labels, word2vec):

#     codevec = [[[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text] for text in texts] for texts in code3d]
        
#     max_sent_len = min([len(sent) for sent in (codevec)],45000)

#     features = pad_code(codevec,max_sent_len) # actually 555 can be any number

#     tensor_data = TensorDataset(torch.tensor(features), torch.from_numpy(encoded_labels))

#     dl = DataLoader(tensor_data, shuffle=True, batch_size=batch_size,drop_last=True)

#     return dl

def pad_features(codevec, padding_idx, seq_length):
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
            features[i, :] = row + [padding_idx]* (seq_length - len(row))
    
    return features

def get_code_str(code, to_lowercase):
    '''
        input
            code (list): a list of code lines from dataset
            to_lowercase (bool)
        output
            code_str: a code in string format
    '''

    code_str = '\n'.join(code)

    if to_lowercase:
        code_str = code_str.lower()

    return code_str

def get_code_vec(code, w2v_model):
    '''
        input
            code (list): a list of code string (from prepare_data_for_LSTM())
            w2v_model (Word2Vec)
        output
            codevec (list): a list of token index of each file
    '''
    codevec = []

    for c in code:
        codevec.append([w2v_model.wv.vocab[word].index if word in w2v_model.wv.vocab else len(w2v_model.wv.vocab) for word in c.split()])

    return codevec

def get_dataloader_for_LSTM(w2v_model, code,encoded_labels, padding_idx):
    '''
        input
            w2v_model (Word2Vec)
            code (list of string)
            encoded_labels (list)
        output

    '''
    codevec = get_code_vec(code, w2v_model)

    # to prevent out of memory error
    max_seq_len = min(max([len(cv) for cv in codevec]),45000)
        
    features = pad_features(codevec, padding_idx, seq_length=max_seq_len)
    tensor_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(np.array(encoded_labels).astype(int)))
    dl = DataLoader(tensor_data, shuffle=True, batch_size=batch_size,drop_last=True)

    return dl

def prepare_data_for_LSTM(df, to_lowercase = False):
    '''
        input
            df (DataFrame): input data from get_df() function
        output
            all_code_str (list): a list of source code in string format
            all_file_label (list): a list of label
    '''
    all_code_str = []
    all_file_label = []

    for filename, group_df in df.groupby('filename'):
        # print(filename)
        # print(group_df)

        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        code_str = get_code_str(code, to_lowercase)

        # if to_lowercase:
        #     code_str = code_str.lower()

        all_code_str.append(code_str)

        all_file_label.append(file_label)

    return all_code_str, all_file_label

def train_model(dataset_name):

    loss_dir = '../../output/loss/CNN/'+dir_suffix+'/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    w2v_dir = get_w2v_path(include_comment=include_comment,include_test_file=include_test_file)
    # w2v_dir = '../'+w2v_dir
    w2v_dir = os.path.join('../'+w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    train_df = get_df(train_rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line,is_baseline=True)
    
    valid_df = get_df(valid_rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line,is_baseline=True)

    # train_code3d, train_label = get_code3d_and_label(train_df, to_lowercase)
    # valid_code3d, valid_label = get_code3d_and_label(valid_df, to_lowercase)

    word2vec_model = Word2Vec.load(w2v_dir)
    
    vocab_size = len(word2vec_model.wv.vocab)  + 1 # for unknown tokens

    # x_train_vec = get_x_vec(train_code3d, word2vec_model)
    # x_valid_vec = get_x_vec(valid_code3d, word2vec_model)

    # max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)

    # train_dl = get_dataloader_for_CNN(train_code3d, train_label, word2vec_model)
    # train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len)

    # valid_dl = get_dataloader(x_valid_vec, valid_label,batch_size,max_sent_len)

    train_code, train_label = prepare_data_for_LSTM(train_df, to_lowercase = to_lowercase)
    valid_code, valid_label = prepare_data_for_LSTM(valid_df, to_lowercase = to_lowercase)

    word2vec_model = Word2Vec.load(w2v_dir)

    padding_idx = word2vec_model.wv.vocab['<pad>'].index

    vocab_size = len(word2vec_model.wv.vocab)+1
        
    train_dl = get_dataloader_for_LSTM(word2vec_model, train_code,train_label, padding_idx)
    valid_dl = get_dataloader_for_LSTM(word2vec_model, valid_code,valid_label, padding_idx)

    net = CNN(batch_size, 1, 1, n_filters, kernel_size, 0.5, vocab_size, embed_dim)

    net = net.cuda()
    
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    criterion = nn.BCELoss()

    checkpoint_files = os.listdir(actual_save_model_dir)

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')

    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:
        word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim)
        net.word_embeddings.weight = nn.Parameter(word2vec_weights, requires_grad=False)

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
    
        # break

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(loss_dir+dataset_name+'-CNN-loss_record.csv',index=False)

    print('finished training model of',dataset_name)


# epoch (int): which epoch to load model
def predict_defective_files_in_releases(dataset_name, target_epochs = 100):
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    w2v_dir = get_w2v_path(include_comment=include_comment,include_test_file=include_test_file)
        # w2v_dir = '../'+w2v_dir
    w2v_dir = os.path.join('../'+w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    train_rel = all_train_releases[dataset_name]
    eval_rels = all_eval_releases[dataset_name][1:]

    word2vec_model = Word2Vec.load(w2v_dir)
    
    vocab_size = len(word2vec_model.wv.vocab) + 1

    net = CNN(batch_size, 1, 1, n_filters, kernel_size, 0.5, vocab_size, embed_dim)

    checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+target_epochs+'epochs.pth')

    net.load_state_dict(checkpoint['model_state_dict'])

    net = net.cuda()
    
    net.eval()
    
    for rel in eval_rels:
        row_list = []

        test_df = get_df(rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line, is_baseline=True)

        for filename, df in tqdm(test_df.groupby('filename')):

            file_label = bool(df['file-label'].unique())

            code = list(df['code_line'])

            code_str = get_code_str(code, to_lowercase)
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
