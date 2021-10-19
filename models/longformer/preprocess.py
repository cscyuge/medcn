import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import os


def tokenize(tokenizer, txt):
    src = re.sub('\*\*', '', txt).lower()
    tokens = tokenizer.tokenize(src)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def preprocess(path_from, path_to):
    with open(os.path.join(path_from,'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    src_all = [u[0] for u in dataset]
    tar_all = [u[1] for u in dataset]

    bert_model = 'schen/longformer-chinese-base-4096'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    src_ids = [tokenize(tokenizer, u) for u in src_all]
    tar_ids = [tokenize(tokenizer, u) for u in tar_all]
    print(len(src_ids))
    src_ids_smaller = []
    tar_ids_smaller = []
    tar_txts = []
    max_len = 512
    for src, tar, txt in zip(src_ids, tar_ids, tar_all):
        if len(src) < max_len and len(tar) < max_len and len(src) > 2 and len(tar) > 2:
            src_ids_smaller.append(src)
            tar_ids_smaller.append(tar)
            tar_txts.append(txt)
    src_ids = src_ids_smaller
    tar_ids = tar_ids_smaller
    print(len(src_ids))
    src_ids = np.array(src_ids)
    tar_ids = np.array(tar_ids)
    length = []
    for src in src_ids:
        length.append(len(src))
    arg_index = np.argsort(length)
    src_ids_unpad = src_ids[arg_index]
    tar_ids_unpad = tar_ids[arg_index]
    src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    tar_ids = pad_sequences(tar_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")

    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]
    tar_masks = [[float(i != 0.0) for i in ii] for ii in tar_ids]
    src_ids = {'pad': src_ids, 'unpad': src_ids_unpad}
    tar_ids = {'pad': tar_ids, 'unpad': tar_ids_unpad}
    with open(os.path.join(path_to, 'src_ids.pkl'), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path_to, 'tar_ids.pkl'), 'wb') as f:
        pickle.dump(tar_ids, f)
    with open(os.path.join(path_to, 'src_masks.pkl'), 'wb') as f:
        pickle.dump(src_masks, f)
    with open(os.path.join(path_to, 'tar_masks.pkl'), 'wb') as f:
        pickle.dump(tar_masks, f)
    with open(os.path.join(path_to, 'tar_txts.pkl'), 'wb') as f:
        pickle.dump(tar_txts, f)


def main():
    print('train dataset:')
    preprocess('../../data/train', './data/train')
    print('test dataset:')
    preprocess('../../data/test', './data/test')
    print('valid dataset:')
    preprocess('../../data/valid', './data/valid')

if __name__ == '__main__':
    main()
    # with open('./data/train/src_ids.pkl','rb') as f:
    #     src_ids = pickle.load(f)
    # print(len(src_ids['pad']))


