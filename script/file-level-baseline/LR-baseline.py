import re, os, pickle, warnings, sys, argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append('../')

from my_util import *

from tqdm import tqdm

warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train',action='store_true')
arg.add_argument('-predict',action='store_true')

args = arg.parse_args()


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

save_model_dir = '../../output/model/LR/'
save_prediction_dir = '../../output/prediction/LR/'

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)

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

# train_release is str
def train_model(dataset_name):
    train_rel = all_train_releases[dataset_name]
    train_df = get_df(train_rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line,is_baseline=True)

    train_code, train_label = prepare_data_for_LSTM(train_df, to_lowercase)

    vectorizer = CountVectorizer() 
    vectorizer.fit(train_code)
    X = vectorizer.transform(train_code).toarray() 
    # Y = list(train_label)
    Y = np.array([1 if label == True else 0 for label in train_label])
    # Y = train_label

    # train_feature = pd.DataFrame(X)
    # train_feature.columns = vectorizer.get_feature_names()
    
    # clf = RandomForestClassifier(random_state = 42, n_jobs=-1)
    clf = LogisticRegression(solver='liblinear')

    clf.fit(X, Y)
    # clf.fit(train_feature, Y)
    
    pickle.dump(clf,open(save_model_dir+re.sub('-.*','',train_rel)+"-LR-model.bin",'wb'))
    pickle.dump(vectorizer,open(save_model_dir+re.sub('-.*','',train_rel)+"-vectorizer.bin",'wb'))
    
    print('finished training model for',dataset_name)

    count_vec_df = pd.DataFrame(X)
    count_vec_df.columns = vectorizer.get_feature_names()

    count_vec_df.to_csv('../../output/count_vec_df/'+train_rel+'.csv',index=False)
    
    # return clf, vectorizer

# test_release is str
def predict_defective_files_in_releases(dataset_name):
    train_release = all_train_releases[dataset_name]
    eval_releases = all_eval_releases[dataset_name][1:]

    clf = pickle.load(open(save_model_dir+re.sub('-.*','',train_release)+"-LR-model.bin",'rb'))
    vectorizer = pickle.load(open(save_model_dir+re.sub('-.*','',train_release)+"-vectorizer.bin",'rb'))

    for rel in eval_releases:
        row_list = []

        test_df = get_df(rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line,is_baseline=True)

        test_code, train_label = prepare_data_for_LSTM(test_df, to_lowercase)

        X = vectorizer.transform(test_code).toarray() 

        # test_feature = pd.DataFrame(X)
        # test_feature.columns = vectorizer.get_feature_names()

        # Y_pred = list(map(bool,list(clf.predict(test_feature))))
        # Y_prob = clf.predict_proba(test_feature)
        Y_pred = list(map(bool,list(clf.predict(X))))
        Y_prob = clf.predict_proba(X)
        Y_prob = list(Y_prob[:,1])

        result_df = pd.DataFrame()
        result_df['project'] = [dataset_name]*len(Y_pred)
        result_df['train'] = [train_release]*len(Y_pred)
        result_df['test'] = [rel]*len(Y_pred)
        result_df['file-level-ground-truth'] = train_label
        result_df['prediction-prob'] = Y_prob
        result_df['prediction-label'] = Y_pred

        result_df.to_csv(save_prediction_dir+rel+'.csv', index=False)


        # c = 0

        # all_arr = []

        # for filename, df in tqdm(test_df.groupby('filename')):

        #     file_label = bool(df['file-label'].unique())

        #     code = list(df['code_line'])

        #     code_str = get_code_str(code, to_lowercase)

        #     X_test = vectorizer.transform([code_str]).toarray()

        #     test_feature = pd.DataFrame(X_test)
        #     test_feature.columns = vectorizer.get_feature_names()

        #     # c = c+1

        #     # break

        #     all_arr.append(X_test[0])

        #     # if c >= 10:
        #     #     break

        #     Y_pred = bool(clf.predict(test_feature))
        #     Y_prob = clf.predict_proba(test_feature)
        #     Y_prob = float(Y_prob[:,1])

        #     row_dict = {
        #                 'project': dataset_name, 
        #                 'train': train_release, 
        #                 'test': rel, 
        #                 'filename': filename, 
        #                 'file-level-ground-truth': file_label, 
        #                 'prediction-prob': Y_prob, 
        #                 'prediction-label': Y_pred
        #                 }
        #     row_list.append(row_dict)

        # df = pd.DataFrame(row_list)
        # df.to_csv(save_prediction_dir+rel+'.csv', index=False)

        # numpy_arr = np.array(all_arr)
        # count_vec_df = pd.DataFrame(numpy_arr)
        # count_vec_df.columns = vectorizer.get_feature_names()

        # count_vec_df.to_csv('../../output/count_vec_df/'+rel+'.csv',index=False)
        print('finish',rel)

        #     break

        # break

        # X_test = vectorizer.transform(test_df['Code']).toarray() # check this input
        # Y_test = list(test_df['Bug'])
        # Y_test = np.array([1 if label == True else 0 for label in Y_test])
        
        # Y_pred = clf.predict(X_test)
        # Y_prob = clf.predict_proba(X_test)
        # Y_prob = Y_prob[:,1]
        
        # prediction_df = pd.DataFrame()
        # prediction_df['train_release'] = [train_release]*len(Y_test)
        # prediction_df['test_release'] = [rel]*len(Y_test)
        # prediction_df['actual'] = Y_test
        # prediction_df['pred'] = Y_pred
        # prediction_df['prob'] = Y_prob

        # prediction_df.to_csv(save_prediction_dir+rel+'.csv',index=False)

        # print('finished predicting',rel)
    
proj_name = args.dataset

if args.train:
    train_model(proj_name)

if args.predict:
    predict_defective_files_in_releases(proj_name)
