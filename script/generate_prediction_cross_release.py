import os, re, argparse, pickle

import torch.optim as optim

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

from DeepLineDP_model import *
from my_util import *

torch.manual_seed(0) 


all_eval_rels_cross_release = {
    'activemq': ['camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'camel': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'derby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'groovy': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'hbase': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'hive': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'jruby': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3'], 
    'lucene': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'wicket-1.5.3'], 
    'wicket': ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 'camel-2.10.0', 'camel-2.11.0', 'derby-10.5.1.1', 'groovy-1_6_BETA_2', 'hbase-0.95.2', 'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1', 'lucene-3.0.0', 'lucene-3.1']}


arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size')
arg.add_argument('-word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
arg.add_argument('-sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
arg.add_argument('-word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
arg.add_argument('-sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
arg.add_argument('-exp_name',type=str,default='')
arg.add_argument('-target_epochs',type=str,default='100')
arg.add_argument('-dropout', type=float, default=0.5, help='dropout rate')
arg.add_argument('-include_comment',action='store_true')
arg.add_argument('-include_blank_line',action='store_true')
arg.add_argument('-include_test_file',action='store_true')

args = arg.parse_args()

weight_dict = {}

# model setting
# batch_size = args.batch_size
# num_epochs = args.num_epochs
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
# lr = args.lr

save_every_epochs = 5
exp_name = args.exp_name

include_comment = args.include_comment
include_blank_line = args.include_blank_line
include_test_file = args.include_test_file

to_lowercase = True

# dir_suffix = 'no-abs-rebalancing-adaptive-ratio2-with-comment'
# dir_suffix = 'rebalancing-adaptive-ratio2'

# dir_suffix = 'correct_prob2'

# dir_suffix = 'rebalancing-adaptive-ratio2-lowercase-new-lr2'
# dir_suffix = 'rebalancing-adaptive-ratio2-lowercase'
dir_suffix = 'no-rebalancing-adaptive-ratio2-lowercase-for-cross-project'

# dir_suffix = 'train-test-subsequent-release'

if include_comment:
    dir_suffix = dir_suffix + '-with-comment'

if include_blank_line:
    dir_suffix = dir_suffix + '-with-blank-line'

if include_test_file:
    dir_suffix = dir_suffix + '-with-test-file'

dir_suffix = dir_suffix+'-'+str(embed_dim)+'-dim'

save_model_dir = '../output/model/DeepLineDP/'+dir_suffix+'/'

dir_suffix = 'cross-release-'+dir_suffix 

intermediate_output_dir = '../output/intermediate_output/DeepLineDP/'+dir_suffix+'/'
# save_model_dir = '../output/model/DeepLineDP/'+dir_suffix+'/'
prediction_dir = '../output/prediction/DeepLineDP/'+dir_suffix+'/'

file_lvl_gt = '../datasets/preprocessed_data/'


if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

def predict_defective_files_in_releases(dataset_name, target_epochs):

    
    actual_save_model_dir = save_model_dir+dataset_name+'/'
    actual_prediction_dir = prediction_dir+dataset_name+'/'
    # actual_intermediate_output_dir = intermediate_output_dir+dataset_name+'/'

    if not os.path.exists(actual_prediction_dir):
        os.makedirs(actual_prediction_dir)

    # if not os.path.exists(actual_intermediate_output_dir):
    #     os.makedirs(actual_intermediate_output_dir)

    train_rel = all_train_releases[dataset_name]
    # test_rel = all_eval_releases[dataset_name][1:]
    # test_rel = [all_eval_releases[dataset_name][0]] # will remove later...
    test_rel = all_eval_rels_cross_release[dataset_name] # not include validation set


    w2v_dir = get_w2v_path(include_comment=include_comment,include_test_file=include_test_file)

    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
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

    for rel in test_rel:
        print('using model from {} to generate prediction of {}'.format(train_rel,rel))
        # print('generating prediction of release:', rel)
        
        actual_intermediate_output_dir = intermediate_output_dir+dataset_name+'/'+train_rel+'-'+rel+'/'

        if not os.path.exists(actual_intermediate_output_dir):
            os.makedirs(actual_intermediate_output_dir)

        test_df = get_df(rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line)
    
        row_list = [] # for creating dataframe later...

        for filename, df in tqdm(test_df.groupby('filename')):

            file_label = bool(df['file-label'].unique())
            line_label = df['line-label'].tolist()
            line_number = df['line_number'].tolist()
            is_comments = df['is_comment'].tolist()

            code = df['code_line'].tolist()

            code2d = prepare_code2d(code, to_lowercase)

            code3d = [code2d]

            codevec = get_x_vec(code3d, word2vec)
            codevec_padded = pad_code(codevec,max_sent_len,limit_sent_len=False, mode='test') 

            save_file_path = actual_intermediate_output_dir+filename.replace('/','_').replace('.java','')+'_'+target_epochs+'_epochs.pkl'
            
            with torch.no_grad():
                codevec_padded_tensor = torch.tensor(codevec_padded)
                output, word_att_weights, line_att_weight = model(codevec_padded_tensor)
                file_prob = output.item()
                prediction = bool(round(output.item()))

            # if not os.path.exists(save_file_path):
            #     with torch.no_grad():
            #         codevec_padded_tensor = torch.tensor(codevec_padded)
            #         output, word_att_weights, line_att_weight = model(codevec_padded_tensor)
            #         file_prob = output.item()
            #         prediction = bool(round(output.item()))

            #         torch.cuda.empty_cache()
                    
            #         output_dict = {
            #             'filename': filename,
            #             'file-label': file_label,
            #             'prob': file_prob,
            #             'pred': prediction,
            #             'word_attention_mat': word_att_weights,
            #             'line_attention_mat': line_att_weight,
            #             'line-label': line_label,
            #             'line-number': line_number
            #         }

            #         pickle.dump(output_dict, open(save_file_path, 'wb'))
                    
            # else:
            #     output_dict = pickle.load(open(save_file_path, 'rb'))
            #     file_prob = output_dict['prob']
            #     prediction = output_dict['pred']
            #     word_att_weights = output_dict['word_attention_mat']
            #     line_att_weight = output_dict['line_attention_mat']

            # break

            numpy_word_attn = word_att_weights[0].cpu().detach().numpy()
            numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

            # loop through each row (based on code)
            # for each row, split list then loop through the list
            # then use the index to get attention from word_att_weights
            # then store in list (somehow...) 

            # Line-level CSV: project, train, test, filename, file-level ground-truth, deeplinedp_prob, line-number, line-level ground-truth, token, attention score

            # print(numpy_word_attn.shape, len(code), len(code2d), len(codevec[0]), len(codevec_padded[0]), codevec_padded_tensor.shape)

            # total_line = min(len(code), numpy_word_attn.shape[0]) 

            for i in range(0,len(code)):
                cur_line = code[i]
                cur_line_label = line_label[i]
                cur_line_number = line_number[i]
                cur_is_comment = is_comments[i]
                cur_line_attn = numpy_line_attn[i]

                token_list = cur_line.strip().split()

                # print(cur_line)

                max_len = min(len(token_list),50) # limit max token each line

                for j in range(0,max_len):  
                    tok = token_list[j]
                    word_attn = numpy_word_attn[i][j]

                    row_dict = {
                        'project': dataset_name, 
                        'train': train_rel, 
                        'test': rel, 
                        'filename': filename, 
                        'file-level-ground-truth': file_label, 
                        'prediction-prob': file_prob, 
                        'prediction-label': prediction, 
                        'line-number': cur_line_number, 
                        'line-level-ground-truth': cur_line_label, 
                        'is-comment-line': cur_is_comment, 
                        'token': tok, 
                        'token-attention-score': word_attn,
                        'line-attention-score': cur_line_attn
                        }

                    row_list.append(row_dict)

                # break

            # break

        df = pd.DataFrame(row_list)

        df.to_csv(actual_prediction_dir+train_rel+'-'+rel+'-'+target_epochs+'-epochs.csv', index=False)

        print('finished release', rel)
        # break


        # if exp_name == '':
        #     prediction_df.to_csv(prediction_dir+rel+'_'+target_epochs+'_epochs.csv',index=False)
        # else:
        #     prediction_df.to_csv(prediction_dir+rel+'_'+exp_name+'_'+target_epochs+'_epochs.csv',index=False)

        # print('finished predicting defective files in',rel)

dataset_name = args.dataset
target_epochs = args.target_epochs

predict_defective_files_in_releases(dataset_name, target_epochs)