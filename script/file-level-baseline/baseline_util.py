
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

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

def prepare_data(df, to_lowercase = False):
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

        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        code_str = get_code_str(code,to_lowercase)

        all_code_str.append(code_str)

        all_file_label.append(file_label)

    return all_code_str, all_file_label

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

def pad_features(codevec, padding_idx, seq_length):
    '''
        input
            codevec (list): a list from get_code_vec()
            padding_idx (int): value used for padding
            seq_length (int): max sequence length of each code line 
    '''
    features = np.zeros((len(codevec), seq_length), dtype=int)
    
    for i, row in enumerate(codevec):
        if len(row) > seq_length:
            features[i,:] = row[:seq_length]
        else:
            features[i, :] = row + [padding_idx]* (seq_length - len(row))
    
    return features
    
def get_dataloader(w2v_model, code,encoded_labels, padding_idx, batch_size):
    '''
        input
            w2v_model (Word2Vec)
            code (list of string)
            encoded_labels (list)
        output
            dataloader object

    '''
    codevec = get_code_vec(code, w2v_model)

    # to prevent out of memory error
    max_seq_len = min(max([len(cv) for cv in codevec]),45000)
        
    features = pad_features(codevec, padding_idx, seq_length=max_seq_len)
    tensor_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(np.array(encoded_labels).astype(int)))
    dl = DataLoader(tensor_data, shuffle=True, batch_size=batch_size,drop_last=True)

    return dl