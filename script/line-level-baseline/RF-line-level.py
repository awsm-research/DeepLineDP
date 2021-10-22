from sklearn.ensemble import RandomForestClassifier

import os, sys, pickle

import numpy as np

from gensim.models import Word2Vec
from torch.utils import data

from tqdm import tqdm

sys.path.append('../')
from DeepLineDP_model import *
from my_util import *

model_dir = '../../output/model/RF-line-level/'
result_dir = '../../output/RF-line-level-result/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

max_grad_norm = 5
embed_dim = 50
word_gru_hidden_dim = 64
sent_gru_hidden_dim = 64
word_gru_num_layers = 1
sent_gru_num_layers = 1
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True

to_lowercase = True

def get_DeepLineDP_and_W2V(dataset_name):
    w2v_dir = get_w2v_path()

    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load('../'+word2vec_file_dir)
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
        dropout=0.001)

    checkpoint = torch.load('../../output/model/DeepLineDP/'+dataset_name+'/checkpoint_7epochs.pth')


    model.load_state_dict(checkpoint['model_state_dict'])

    model.sent_attention.word_attention.freeze_embeddings(True)

    model = model.cuda()
    model.eval()

    return model, word2vec

def train_RF_model(dataset_name):
    model, word2vec = get_DeepLineDP_and_W2V(dataset_name)

    clf = RandomForestClassifier(random_state=0, n_jobs=24)

    train_rel = all_train_releases[dataset_name]

    train_df = get_df(train_rel, is_baseline=True)
    
    line_rep_list = []
    all_line_label = []

    # loop to get line representation of each file in train data
    for filename, df in tqdm(train_df.groupby('filename')):

        code = df['code_line'].tolist()
        line_label = df['line-label'].tolist()

        all_line_label.extend(line_label)

        code2d = prepare_code2d(code, to_lowercase)

        code3d = [code2d]

        codevec = get_x_vec(code3d, word2vec)

        with torch.no_grad():
            codevec_padded_tensor = torch.tensor(codevec)
            _, __, ___, line_rep = model(codevec_padded_tensor)

        numpy_line_rep = line_rep.cpu().detach().numpy()

        line_rep_list.append(numpy_line_rep)

    x = np.concatenate(line_rep_list)

    print('prepare data finished')

    clf.fit(x,all_line_label)

    pickle.dump(clf, open(model_dir+dataset_name+'-RF-model.bin','wb'))

    print('finished training model of',dataset_name)


def predict_defective_line(dataset_name):
    model, word2vec = get_DeepLineDP_and_W2V(dataset_name)
    clf = pickle.load(open(model_dir+dataset_name+'-RF-model.bin','rb'))

    print('load model finished')

    test_rels = all_eval_releases[dataset_name][1:]

    for rel in test_rels:
        test_df = get_df(rel, is_baseline=True)

        test_df = test_df[test_df['file-label']==True]
        test_df = test_df.drop(['is_comment','is_test_file','is_blank'],axis=1)

        all_df_list = [] # store df for saving later...

        for filename, df in tqdm(test_df.groupby('filename')):

            code = df['code_line'].tolist()

            code2d = prepare_code2d(code, to_lowercase)

            code3d = [code2d]

            codevec = get_x_vec(code3d, word2vec)

            with torch.no_grad():
                codevec_padded_tensor = torch.tensor(codevec)
                _, __, ___, line_rep = model(codevec_padded_tensor)

            numpy_line_rep = line_rep.cpu().detach().numpy()

            pred = clf.predict(numpy_line_rep)

            df['line-score-pred'] = pred.astype(int)

            all_df_list.append(df)

        all_df = pd.concat(all_df_list)

        all_df.to_csv(result_dir+rel+'-line-lvl-result.csv',index=False)

        print('finished',rel)

proj_name = sys.argv[1]

train_RF_model(proj_name)
predict_defective_line(proj_name)
