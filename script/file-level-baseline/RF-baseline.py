import re, os, pickle, warnings, sys, argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

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

save_model_dir = '../../output/model/RF/'
save_prediction_dir = '../../output/prediction/RF/'

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
    Y = list(train_label)
    Y = np.array([1 if label == True else 0 for label in Y])
    
    clf = RandomForestClassifier(random_state = 0)
    clf.fit(X, Y)
    
    pickle.dump(clf,open(save_model_dir+re.sub('-.*','',train_rel)+"-RF-model.bin",'wb'))
    pickle.dump(vectorizer,open(save_model_dir+re.sub('-.*','',train_rel)+"-vectorizer.bin",'wb'))
    
    print('finished training model for',dataset_name)
    # return clf, vectorizer

# test_release is str
def predict_defective_files_in_releases(dataset_name):
    train_release = all_train_releases[dataset_name]
    eval_releases = all_eval_releases[dataset_name][1:]

    clf = pickle.load(open(save_model_dir+re.sub('-.*','',train_release)+"-RF-model.bin",'rb'))
    vectorizer = pickle.load(open(save_model_dir+re.sub('-.*','',train_release)+"-vectorizer.bin",'rb'))

    for rel in eval_releases:
        row_list = []

        test_df = get_df(rel, include_comment=include_comment, include_test_files=include_test_file, include_blank_line=include_blank_line,is_baseline=True)

        for filename, df in tqdm(test_df.groupby('filename')):

            file_label = bool(df['file-label'].unique())

            code = list(df['code_line'])

            code_str = get_code_str(code, to_lowercase)

            X_test = vectorizer.transform([code_str]).toarray()

            Y_pred = bool(clf.predict(X_test))
            Y_prob = clf.predict_proba(X_test)
            Y_prob = float(Y_prob[:,1])

            row_dict = {
                        'project': dataset_name, 
                        'train': train_release, 
                        'test': rel, 
                        'filename': filename, 
                        'file-level-ground-truth': file_label, 
                        'prediction-prob': Y_prob, 
                        'prediction-label': Y_pred
                        }
            row_list.append(row_dict)

        df = pd.DataFrame(row_list)
        df.to_csv(save_prediction_dir+rel+'.csv', index=False)

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

        print('finished predicting',rel)
    
proj_name = args.dataset

if args.train:
    train_model(proj_name)

if args.predict:
    predict_defective_files_in_releases(proj_name)
