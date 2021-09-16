import numpy as np
import pandas as pd
import os, pickle, sys, argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


from gensim.models import Word2Vec

from dbn.models import SupervisedDBNClassification

sys.path.append('../')

from my_util import *

arg = argparse.ArgumentParser()
arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train',action='store_true')
arg.add_argument('-predict',action='store_true')

args = arg.parse_args()


'''
    The model setting is based on the paper "Automatically Learning Semantic Features for Defect Prediction"
    code from https://github.com/albertbup/deep-belief-network
'''

# model parameter
batch_size = 30
hidden_layers_structure = [100]*10

save_model_dir = './baseline-model-DBN/'
save_prediction_dir = './prediction-DBN/'

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)

def pad_features(codevec, seq_length):
    ''' padded codevec with 0's or truncated to the input seq_length.
    '''
    ## getting the correct rows x cols shape
    features = np.zeros((len(codevec), seq_length), dtype=int)

    for i, row in enumerate(codevec):
        if len(row) > seq_length:
            features[i,:] = row[:seq_length]
        else:
            features[i, :] = row + [0]* (seq_length - len(row))

    return features

def convert_to_token_index(w2v_model, code, max_seq_len):
    codevec = []
    
    for c in code:
        codevec.append([w2v_model.wv.vocab[word].index if word in w2v_model.wv.vocab else len(w2v_model.wv.vocab) for word in c.split()])

    features = pad_features(codevec, seq_length=max_seq_len)
     
    return features

def get_max_code_length(dataset_name, w2v_model):
    train_releases = all_train_releases[dataset_name]
    eval_rel = all_eval_releases[dataset_name][1:]

    train_df = get_df_for_baseline(train_releases)
    code, _ = get_data_and_label(train_df)
    
    codevec = []
    
    for c in code:
        codevec.append([w2v_model.wv.vocab[word].index if word in w2v_model.wv.vocab else len(w2v_model.wv.vocab) for word in c.split()])
            
    max_seq_len = max([len(cv) for cv in codevec])
    
    for rel in eval_rel:
        test_df = get_df_for_baseline(rel)
        code, _ = get_data_and_label(test_df)
        
        codevec = []
    
        for c in code:
            codevec.append([w2v_model.wv.vocab[word].index if word in w2v_model.wv.vocab else len(w2v_model.wv.vocab) for word in c.split()])

        max_seq_len = max([len(cv) for cv in codevec])
    
    if max_seq_len > 125000:
        max_seq_len = 125000
        
    return max_seq_len

def train_model(dataset_name):
    train_releases = all_train_releases[dataset_name]
    
    train_df = get_df_for_baseline(train_releases)
    code,encoded_labels = get_data_and_label(train_df)

    word2vec_file_dir = os.path.join('.'+word2vec_baseline_file_dir,dataset_name+'.bin')

    word2vec_model = Word2Vec.load(word2vec_file_dir)
        
    max_seq_len = get_max_code_length(dataset_name, word2vec_model)
    
    token_idx = convert_to_token_index(word2vec_model, code, max_seq_len)

    scaler = MinMaxScaler(feature_range=(0,1))
    features = scaler.fit_transform(token_idx)
    
    dbn_clf = SupervisedDBNClassification(hidden_layers_structure=hidden_layers_structure,
             learning_rate_rbm=0.01,
             learning_rate=0.1,
             n_epochs_rbm=200,    # default is 200
             n_iter_backprop=200, # default is 200
             batch_size=batch_size,
             activation_function='sigmoid')
    
    dbn_clf.fit(features,encoded_labels)
    
    dbn_features = dbn_clf.transform(features)
    
    rf_clf = RandomForestClassifier(n_jobs=24)
    
    rf_clf.fit(dbn_features,encoded_labels)
    
    pickle.dump(dbn_clf,open(save_model_dir+dataset_name+'-DBN.pkl','wb'))
    pickle.dump(rf_clf,open(save_model_dir+dataset_name+'-RF.pkl','wb'))
    
    print('finished training model of',dataset_name)

# epoch (int): which epoch to load model
def predict_defective_files_in_releases(dataset_name):
    train_rel = all_train_releases[dataset_name]
    eval_rel = all_eval_releases[dataset_name][1:]

    word2vec_file_dir = os.path.join(word2vec_baseline_file_dir,dataset_name+'.bin')

    word2vec_model = Word2Vec.load(word2vec_file_dir)
    
    dbn_clf = pickle.load(open(save_model_dir+dataset_name+'-DBN.pkl','rb'))
    rf_clf = pickle.load(open(save_model_dir+dataset_name+'-RF.pkl','rb'))
    
    for rel in eval_rel:

        scaler = MinMaxScaler(feature_range=(0,1))
        
        test_df = get_df_for_baseline(rel)
        code,encoded_labels = get_data_and_label(test_df)
        
        max_seq_len = get_max_code_length(dataset_name, word2vec_model)
    
        token_idx = convert_to_token_index(word2vec_model, code, max_seq_len)
        
        features = scaler.fit_transform(token_idx)
        
        dbn_features = dbn_clf.transform(features)
        
        y_pred = rf_clf.predict(dbn_features)
        y_prob = rf_clf.predict_proba(dbn_features)
        y_prob = y_prob[:,1] 
        
        prediction_df = pd.DataFrame()
        prediction_df['train_release'] = [train_rel]*len(encoded_labels)
        prediction_df['test_release'] = [rel]*len(encoded_labels)
        prediction_df['actual'] = encoded_labels
        prediction_df['pred'] = y_pred
        prediction_df['prob'] = y_prob
        
        print('finished release',rel)

        prediction_df.to_csv(save_prediction_dir+rel+'.csv',index=False)

    print('-'*100,'\n')

proj_name = args.dataset

if args.train:
    train_model(proj_name)

if args.predict:
    predict_defective_files_in_releases(proj_name)

# train_model('activemq') 
# predict_defective_files_in_releases('activemq')
