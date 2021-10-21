import os

import pandas as pd
from tqdm import tqdm

from my_util import *

base_data_dir = '../datasets/preprocessed_data/'
base_original_data_dir = '../datasets/original/File-level/'

data_for_ngram_dir = '../datasets/n_gram_data/'
data_for_error_prone_dir = '../datasets/ErrorProne_data/'

proj_names = list(all_train_releases.keys())

def export_df_to_files(data_df, code_file_dir, line_file_dir):

    for filename, df in tqdm(data_df.groupby('filename')):

        code_lines = list(df['code_line'])
        code_str = '\n'.join(code_lines)
        code_str = code_str.lower()
        line_num = list(df['line_number'])
        line_num = [str(l) for l in line_num]

        code_filename = filename.replace('/','_').replace('.java','')+'.txt'
        line_filename = filename.replace('/','_').replace('.java','')+'_line_num.txt'

        with open(code_file_dir+code_filename,'w') as f:
            f.write(code_str)

        with open(line_file_dir+line_filename, 'w') as f:
            f.write('\n'.join(line_num))

def export_ngram_data_each_release(release, is_train = False):

    file_dir = data_for_ngram_dir+release+'/'
    file_src_dir = file_dir+'src/'
    file_line_num_dir = file_dir+'line_num/'

    if not os.path.exists(file_src_dir):
        os.makedirs(file_src_dir)

    if not os.path.exists(file_line_num_dir):
        os.makedirs(file_line_num_dir)

    data_df = pd.read_csv(base_data_dir+release+'.csv', encoding='latin')

    # get clean files for training only
    if is_train:
        data_df = data_df[(data_df['is_test_file']==False) & (data_df['is_blank']==False) & (data_df['file-label']==False)]
    # get defective files for prediction only
    else:
        data_df = data_df[(data_df['is_test_file']==False) & (data_df['is_blank']==False) & (data_df['file-label']==True)]

    data_df = data_df.fillna('')
    
    export_df_to_files(data_df, file_src_dir, file_line_num_dir)

def export_data_all_releases(proj_name):
    train_rel = all_train_releases[proj_name]
    eval_rels = all_eval_releases[proj_name]

    export_ngram_data_each_release(train_rel, True)

    for rel in eval_rels:
        export_ngram_data_each_release(rel, False)
        # break

def export_ngram_data_all_projs():
    for proj in proj_names:
        export_data_all_releases(proj)
        print('finish',proj)

def export_errorprone_data(proj_name):
    cur_eval_rels = all_eval_releases[proj_name][1:]

    for rel in cur_eval_rels:

        save_dir = data_for_error_prone_dir+rel+'/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_df = pd.read_csv(base_original_data_dir+rel+'_ground-truth-files_dataset.csv', encoding='latin')

        data_df = data_df[data_df['Bug']==True]

        for filename, df in data_df.groupby('File'):

            if 'test' in filename or '.java' not in filename:
                continue

            filename = filename.replace('/','_')

            code = list(df['SRC'])[0].strip()

            with open(save_dir+filename,'w') as f:
                f.write(code)

        print('finish release',rel)


def export_error_prone_data_all_projs():
    for proj in proj_names:
        export_errorprone_data(proj)
        print('finish',proj)

export_ngram_data_all_projs()
export_error_prone_data_all_projs()