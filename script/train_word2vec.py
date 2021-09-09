import os

from gensim.models import Word2Vec

import more_itertools

from DeepLineDP_model import *
from my_util import *

# if not os.path.exists(word2vec_deepline_dp_file_dir):
#     os.makedirs(word2vec_deepline_dp_file_dir)

# if not os.path.exists(word2vec_baseline_file_dir):
#     os.makedirs(word2vec_baseline_file_dir)

def get_w2v_path(include_comment=False,include_test_file=False):
    suffix = ''

    if include_comment:
        suffix = suffix + 'with-comment-'
    else:
        suffix = suffix + 'without-comment-'

    if include_test_file:
        suffix = suffix + 'with-test-file'
    else:
        suffix = suffix + 'without-test-file'

    actual_w2v_dir = word2vec_dir+suffix+'/'

    return actual_w2v_dir

def train_word2vec_model(dataset_name,include_comment=False,include_test_file=False, embedding_dim = 50):

    w2v_path = get_w2v_path(include_comment, include_test_file)

    save_path = w2v_path+'/'+dataset_name+'-'+str(embedding_dim)+'dim.bin'

    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    train_rel = all_train_releases[dataset_name]
    
    train_rel = all_train_releases[dataset_name]

    train_df = get_df(train_rel,include_comment=include_comment, include_test_files=include_test_file)

    train_code_3d, _ = get_code3d_and_label(train_df)

    all_texts = list(more_itertools.collapse(train_code_3d[:],levels=1))

    word2vec = Word2Vec(all_texts,size=embedding_dim, min_count=1,sorted_vocab=1) #size is embedding size

    # word2vec2 = Word2Vec(all_texts,size=100, min_count=1,sorted_vocab=1) #size is embedding size

    print('finish Word2Vec training for',dataset_name)

    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))

    # word2vec2.save(word2vec_baseline_file_dir+'/'+dataset_name+'.bin')

train_word2vec_model('activemq',False,False,50)

# def train_word2vec_model_cross_release(dataset_name):
#     cur_all_rels = all_releases[dataset_name]
    
#     for i in range(0,len(cur_all_rels)-1):
#         train_rel = cur_all_rels[i]
#         train_code_3d, _, __, ___ = prepare_data_cross_release(train_rel)

#         all_texts = list(more_itertools.collapse(train_code_3d[:],levels=1))

#         word2vec1 = Word2Vec(all_texts,size=50, min_count=1,sorted_vocab=1) #size is embedding size
#         word2vec2 = Word2Vec(all_texts,size=100, min_count=1,sorted_vocab=1) #size is embedding size

#         print('finish Word2Vec training for',train_rel)

#         word2vec1.save(word2vec_deepline_dp_file_dir+'/'+train_rel+'.bin')
#         word2vec2.save(word2vec_baseline_file_dir+'/'+train_rel+'.bin')

# for p in all_projs:
#     train_word2vec_model_cross_release(p)