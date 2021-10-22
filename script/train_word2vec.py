import os, sys

from gensim.models import Word2Vec

import more_itertools

from DeepLineDP_model import *
from my_util import *


def train_word2vec_model(dataset_name, embedding_dim = 50):

    w2v_path = get_w2v_path()

    save_path = w2v_path+'/'+dataset_name+'-'+str(embedding_dim)+'dim.bin'

    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    train_rel = all_train_releases[dataset_name]

    train_df = get_df(train_rel)

    train_code_3d, _ = get_code3d_and_label(train_df, True)

    all_texts = list(more_itertools.collapse(train_code_3d[:],levels=1))

    word2vec = Word2Vec(all_texts,size=embedding_dim, min_count=1,sorted_vocab=1)

    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))


p = sys.argv[1]

train_word2vec_model(p,50)
