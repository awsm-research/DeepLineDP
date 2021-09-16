# run passed

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

import pandas as pd
import os, sys ,argparse

from gensim.models import Word2Vec

from tqdm import tqdm

sys.path.append('../')

from my_util import *

arg = argparse.ArgumentParser()
arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-epochs', type=int, default=30)
arg.add_argument('-target_epochs', type=str, default=100, help='which epoch of model to load')
arg.add_argument('-train',action='store_true')
arg.add_argument('-predict',action='store_true')

args = arg.parse_args()


# model parameters

batch_size = 8
seq_length = 7000 # max length of all code in the whole dataset
output_size = 1
embedding_dim = 100
n_filters = 100
kernel_size = [5,5] # for 2 conv layers
lr = 0.001

criterion = nn.BCELoss()

epochs = args.epochs # default is 100
save_every_epochs = 5 # default is 5
max_seq_len = 50 # number of tokens of each line in a file

save_model_dir = './baseline-model-CNN/'
save_prediction_dir = './prediction-CNN/'

if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)

if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
    
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

        all_out = torch.cat((max_out1, max_out2), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.fc(fc_in)
        sig_out = self.sig(logits)
        return sig_out
                                            

def pad_code(code_list_3d,max_sent_len):
    paded = []
    
    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            else:
                new_line = line+[0]*(max_seq_len - len(line))
            sent_list.extend(new_line)
            
        if max_sent_len-len(file) > 0:
            for i in range(0,max_sent_len-len(file)):
                sent_list.extend([0]*max_seq_len)
                
        paded.append(sent_list[:max_sent_len])

    return paded


def get_dataloader_for_CNN(word2vec, code,encoded_labels):
    
    code3d = create3DList(code)
    
    codevec = [[[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text] for text in texts] for texts in code3d]
        
    max_sent_len = max([len(sent) for sent in (codevec)])
    features = pad_code(codevec,max_sent_len) # actually 555 can be any number

    tensor_data = TensorDataset(torch.tensor(features), torch.from_numpy(encoded_labels))
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
        
    train_dl = get_dataloader_for_CNN(word2vec_model, train_code, train_encoded_labels)
    valid_dl = get_dataloader_for_CNN(word2vec_model, valid_code, valid_encoded_labels)

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec_model, embedding_dim)
    
    net = CNN(batch_size, 1, 1, n_filters, kernel_size, 0.5, vocab_size, embedding_dim)

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


# epoch (int): which epoch to load model
def predict_defective_files_in_releases(dataset_name, target_epochs = 100):

    actual_save_model_dir = save_model_dir+dataset_name+'/'

    train_rel = all_train_releases[dataset_name]
    eval_rel = all_eval_releases[dataset_name][1:]

    word2vec_file_dir = os.path.join('.'+word2vec_baseline_file_dir,dataset_name+'.bin')

    word2vec_model = Word2Vec.load(word2vec_file_dir)
    
    vocab_size = len(word2vec_model.wv.vocab) + 1

    net = CNN(batch_size, 1, 1, n_filters, kernel_size, 0.5, vocab_size, embedding_dim)

    checkpoint = torch.load(actual_save_model_dir+'checkpoint_'+target_epochs+'epochs.pth')

    net.load_state_dict(checkpoint['model_state_dict'])

    net = net.cuda()

    net.eval()

    for rel in eval_rel:
        test_df = get_df_for_baseline(rel)
        code,encoded_labels = get_data_and_label(test_df)
        test_dl = get_dataloader_for_CNN(word2vec_model, code,encoded_labels)

        y_pred = []
        y_test = []
        y_prob = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_dl):
                inputs, labels = inputs.cuda(), labels.cuda()
                output = net(inputs)

                pred = torch.round(output)
                
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
