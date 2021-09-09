import re, os

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from sklearn.utils import shuffle

max_seq_len = 50

all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6', 
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0', 
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}


all_projs = list(all_train_releases.keys())

# file_lvl_gt = os.path.join(os.path.dirname(os.getcwd()),'java-parsed-AST-no-comment-no-varname-no-methodname/')
file_lvl_gt = '../datasets/parsed_dataset_with_comments/'
# file_lvl_gt = '../datasets/parsed_dataset_without_comments/'
# file_lvl_gt = '../datasets/abstract_code_without_comments/'

file_lvl_gt_for_baseline = '../../datasets/parsed_dataset_without_comments/'

# file_lvl_gt_for_baseline = '../../datasets/java-parsed-AST-no-comment-no-varname-no-methodname/'


# word2vec_dir = './Word2Vec_model-50dim-with-comment/'
word2vec_dir = './Word2Vec_model-50dim-cross-release/'
word2vec_deepline_dp_file_dir = os.path.join(word2vec_dir,'DeepLineDP')
word2vec_baseline_file_dir = os.path.join(word2vec_dir,'baseline')
# word2vec_deepline_dp_file_dir = os.path.join(word2vec_dir,'DeepLineDP_abs')
# word2vec_baseline_file_dir = os.path.join(word2vec_dir,'baseline_abs')

loss_dir = './loss/'

def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0).cuda()
    
    # for add zero vector for unknown tokens
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1,embed_dim).cuda()))

    return word2vec_weights

def get_actual_code_without_comment(java_code):
    # return list of line numbers, code without comment (in str format)
    comment_pattern = '//.*?\n|/\*.*?\*/\n'
    code_lines = java_code.splitlines() # problem is here (but why?????)
    code_lines_with_line_num = []

    for i in range(0,len(code_lines)):
        tmp_line = str(code_lines[i])
        if tmp_line.endswith('*/'):
            code_lines_with_line_num.append(tmp_line.replace('*/','')+' <LINE '+ str(i+1) + '>*/')
        else:
            code_lines_with_line_num.append(tmp_line+' <LINE '+ str(i+1) + '>')

    code_lines_with_line_num = '\n'.join(code_lines_with_line_num)

    no_comment_code = re.sub(comment_pattern, '', code_lines_with_line_num, flags=re.S).replace('{',' ').replace('}',' ').replace('(',' ').replace(')',' ')

    line_num_list = re.findall(r'<LINE \d+>', no_comment_code)
    line_num_list = [int(l.replace('<LINE ','').replace('>','')) for l in line_num_list]

    no_comment_code_lines = no_comment_code.split('\n')
    no_comment_code_lines = [s.strip() for s in no_comment_code_lines]

    while '' in no_comment_code_lines:
        no_comment_code_lines.remove('')

    no_comment_code_lines = '\n'.join(no_comment_code_lines)
    no_comment_code_lines = re.sub('<LINE .*>','',no_comment_code_lines)

    return no_comment_code_lines, line_num_list

def create3DList(code_list, ignore_blank_line = True):
    code_3d = []
    
    for c in code_list:
        raw_code = c.split('\n')
        token_list = []
        for l in raw_code:    
            l_clean = re.sub('\\s+',' ',l)
            l_clean = l.strip()

            if ignore_blank_line:
                if len(l_clean) < 1: # ignore blank lines
                    continue

            tokens = l_clean.split()
            token_list.append(tokens)
        code_3d.append(token_list)
    
    return code_3d

def pad_code(code_list_3d,max_sent_len,limit_sent_len=True):
    paded = []
    
    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            else:
                new_line = line+[0]*(max_seq_len - len(line))
            sent_list.append(new_line)
            
        if limit_sent_len:
            if max_sent_len-len(file) > 0:
                for i in range(0,max_sent_len-len(file)):
                    sent_list.append([0]*max_seq_len)
                
        paded.append(sent_list[:max_sent_len])
        
    return paded

def get_df_for_baseline(release):

    df = pd.read_csv(file_lvl_gt_for_baseline+release+'.txt',sep='\t')
    # else:
    #     df = pd.read_csv(file_lvl_gt+release+'.txt',sep='\t')
    df = df.dropna()
#     df = df.head() # just for testing
    # source code that has bug comes first
    # sorted_df = df.sort_values('Bug',ascending=False)
    
    return df

def get_data_and_label(df):
    code = list(df['Code'])
#     code = [re.sub('\\s+',' ',c.lower()) for c in code] # make all code to lowercase

    labels = list(df['Bug'])
    encoded_labels = np.array([1 if label == True else 0 for label in labels])
    
    return code,encoded_labels # both are list of code and label

def get_dataloader(code_vec, label_list,batch_size, max_sent_len):
    y_tensor =  torch.cuda.FloatTensor([label for label in label_list])
    code_vec_pad = pad_code(code_vec,max_sent_len)
    tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)
    dl = DataLoader(tensor_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    
    return dl

def prepare_data(rel):
    df = pd.read_csv(file_lvl_gt+rel+'.txt',sep='\t')
    df = df.dropna()
    
    code = list(df['Code'])
    
    code_3D_list = create3DList(code)
    label = list(df['Bug'])
    
    return code_3D_list, label

def prepare_data_cross_release(rel):
    df = pd.read_csv(file_lvl_gt+rel+'.txt',sep='\t')
    df = df.dropna()

    total_samples = len(df)
    train_samples = round(0.9*total_samples)
    # valid_samples = total_samples-train_samples
    
    df_shuffled=shuffle(df, random_state=0)

    code = list(df['Code'])

    train_code = code[:train_samples]
    valid_code = code[train_samples:]
    
    train_code_3D_list = create3DList(train_code)
    valid_code_3D_list = create3DList(valid_code)

    label = list(df['Bug'])

    train_label = label[:train_samples]
    valid_label = label[train_samples:]
    
    return train_code_3D_list, valid_code_3D_list, train_label, valid_label

def get_x_vec(code_3d, word2vec):
    x_vec = [[[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text]
         for text in texts] for texts in code_3d]
    
    return x_vec