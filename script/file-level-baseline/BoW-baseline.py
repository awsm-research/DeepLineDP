import re, os, pickle, warnings, sys, argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from baseline_util import *

sys.path.append('../')

from my_util import *

from imblearn.over_sampling import SMOTE 

warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train',action='store_true')
arg.add_argument('-predict',action='store_true')

args = arg.parse_args()

save_model_dir = '../../output/model/BoW/'
save_prediction_dir = '../../output/prediction/BoW/'

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

if not os.path.exists(save_prediction_dir):
    os.makedirs(save_prediction_dir)


# train_release is str
def train_model(dataset_name):
    train_rel = all_train_releases[dataset_name]
    train_df = get_df(train_rel, is_baseline=True)

    train_code, train_label = prepare_data(train_df, True)

    vectorizer = CountVectorizer() 
    vectorizer.fit(train_code)
    X = vectorizer.transform(train_code).toarray()
    train_feature = pd.DataFrame(X)
    Y = np.array([1 if label == True else 0 for label in train_label])

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(train_feature, Y)

    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_res, y_res)
    
    pickle.dump(clf,open(save_model_dir+re.sub('-.*','',train_rel)+"-BoW-model.bin",'wb'))
    pickle.dump(vectorizer,open(save_model_dir+re.sub('-.*','',train_rel)+"-vectorizer.bin",'wb'))
    
    print('finished training model for',dataset_name)

    count_vec_df = pd.DataFrame(X)
    count_vec_df.columns = vectorizer.get_feature_names()
    

# test_release is str
def predict_defective_files_in_releases(dataset_name):
    train_release = all_train_releases[dataset_name]
    eval_releases = all_eval_releases[dataset_name][1:]

    clf = pickle.load(open(save_model_dir+re.sub('-.*','',train_release)+"-BoW-model.bin",'rb'))
    vectorizer = pickle.load(open(save_model_dir+re.sub('-.*','',train_release)+"-vectorizer.bin",'rb'))

    for rel in eval_releases:

        test_df = get_df(rel,is_baseline=True)

        test_code, train_label = prepare_data(test_df, True)

        X = vectorizer.transform(test_code).toarray() 

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

        print('finish',rel)

proj_name = args.dataset

if args.train:
    train_model(proj_name)

if args.predict:
    predict_defective_files_in_releases(proj_name)
