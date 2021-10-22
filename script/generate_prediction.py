import os, argparse, pickle

import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

from DeepLineDP_model import *
from my_util import *

torch.manual_seed(0)

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size')
arg.add_argument('-word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
arg.add_argument('-sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
arg.add_argument('-word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
arg.add_argument('-sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
arg.add_argument('-exp_name',type=str,default='')
arg.add_argument('-target_epochs',type=str,default='7', help='the epoch to load model')
arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')

args = arg.parse_args()

weight_dict = {}

# model setting
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

save_every_epochs = 5
exp_name = args.exp_name

save_model_dir = '../output/model/DeepLineDP/'
intermediate_output_dir = '../output/intermediate_output/DeepLineDP/within-release/'
prediction_dir = '../output/prediction/DeepLineDP/within-release/'

file_lvl_gt = '../datasets/preprocessed_data/'


if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

def predict_defective_files_in_releases(dataset_name, target_epochs):

    actual_save_model_dir = save_model_dir+dataset_name+'/'

    train_rel = all_train_releases[dataset_name]
    test_rel = all_eval_releases[dataset_name][1:]

    w2v_dir = get_w2v_path()

    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')

    total_vocab = len(word2vec.wv.vocab)

    vocab_size = total_vocab +1 # for unknown tokens
  
    model = HierarchicalAttentionNetwork(
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
        print('generating prediction of release:', rel)
        
        actual_intermediate_output_dir = intermediate_output_dir+rel+'/'

        if not os.path.exists(actual_intermediate_output_dir):
            os.makedirs(actual_intermediate_output_dir)

        test_df = get_df(rel)
    
        row_list = [] # for creating dataframe later...

        for filename, df in tqdm(test_df.groupby('filename')):

            file_label = bool(df['file-label'].unique())
            line_label = df['line-label'].tolist()
            line_number = df['line_number'].tolist()
            is_comments = df['is_comment'].tolist()

            code = df['code_line'].tolist()

            code2d = prepare_code2d(code, True)

            code3d = [code2d]

            codevec = get_x_vec(code3d, word2vec)

            save_file_path = actual_intermediate_output_dir+filename.replace('/','_').replace('.java','')+'_'+target_epochs+'_epochs.pkl'
            
            if not os.path.exists(save_file_path):
                with torch.no_grad():
                    codevec_padded_tensor = torch.tensor(codevec)
                    output, word_att_weights, line_att_weight, _ = model(codevec_padded_tensor)
                    file_prob = output.item()
                    prediction = bool(round(output.item()))

                    torch.cuda.empty_cache()

                    output_dict = {
                        'filename': filename,
                        'file-label': file_label,
                        'prob': file_prob,
                        'pred': prediction,
                        'word_attention_mat': word_att_weights,
                        'line_attention_mat': line_att_weight,
                        'line-label': line_label,
                        'line-number': line_number
                    }

                    pickle.dump(output_dict, open(save_file_path, 'wb'))
                    
            else:
                output_dict = pickle.load(open(save_file_path, 'rb'))
                file_prob = output_dict['prob']
                prediction = output_dict['pred']
                word_att_weights = output_dict['word_attention_mat']
                line_att_weight = output_dict['line_attention_mat']

            numpy_word_attn = word_att_weights[0].cpu().detach().numpy()
            numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

            # for each line in source code
            for i in range(0,len(code)):
                cur_line = code[i]
                cur_line_label = line_label[i]
                cur_line_number = line_number[i]
                cur_is_comment = is_comments[i]
                cur_line_attn = numpy_line_attn[i]

                token_list = cur_line.strip().split()

                max_len = min(len(token_list),50) # limit max token each line

                # for each token in a line
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

        df = pd.DataFrame(row_list)

        df.to_csv(prediction_dir+rel+'.csv', index=False)

        print('finished release', rel)

dataset_name = args.dataset
target_epochs = args.target_epochs

predict_defective_files_in_releases(dataset_name, target_epochs)