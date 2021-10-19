import pickle

import torch
from utils.dataset import build_dataset
from tqdm import tqdm
import copy
from utils.bleu_eval import count_score
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from model import build


def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(8, True)

    save_file_best = torch.load('./cache/best_save.data', map_location={'cuda:0': 'cuda:0'})

    model.load_state_dict(save_file_best['para'])
    model.eval()

    val_data = build_dataset(config, '../longformer/data/valid/src_ids.pkl', '../longformer/data/valid/src_masks.pkl',
                             '../longformer/data/valid/tar_ids.pkl',
                             '../longformer/data/valid/tar_masks.pkl', '../longformer/data/valid/tar_txts.pkl')

    max_len = 64
    outs = []
    tars = []
    for token_ids_src, _, seq_len_src, mask_src, token_ids_tar, __, seq_len_tar, mask_tar, tar_txt in tqdm(val_data):
        src = ''.join(config.tokenizer.convert_ids_to_tokens(token_ids_src))
        src = src.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        tar = []
        datas = []
        src_ids = []
        for sts in src.split('。'):
            # print(sts)
            tokens = config.tokenizer.tokenize(sts)
            tokens = ['▁'] + tokens + ['</s>']
            ids = config.tokenizer.convert_tokens_to_ids(tokens)
            if len(ids) < max_len and len(ids) > 2:
                tar.append('[PAD]')
                src_ids.append(ids)
            else:
                tar.append(sts)
        if len(src_ids) > 0:
            src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=-100, truncating="post", padding="post")
            src_masks = [[float(i != -100) for i in ii] for ii in src_ids]
            seq_len_srcs = len(src_ids[0]) - np.sum(src_ids == -100, axis=1)
            for ids, seq_len_src, masks in zip(src_ids, seq_len_srcs, src_masks):
                datas.append((ids, seq_len_src, masks))

            x_src = torch.LongTensor([_[0] for _ in datas]).to(config.device)
            seq_len = torch.LongTensor([_[1] for _ in datas])
            masks = torch.LongTensor([_[2] for _ in datas]).to(config.device)
            batch_src = (x_src, seq_len, masks)
            with torch.no_grad():
                outputs = model.generate(batch_src[0])
                sentences = [config.tokenizer.decode(u, skip_special_tokens=True) for u in outputs]


            index = 0
            for i, u in enumerate(tar):
                if u == '[PAD]':
                    tar[i] = sentences[index]
                    index += 1

        tar = '。'.join(tar)
        outs.append(tar)
        tars.append(tar_txt)

    with open('./result/paragraph_out.txt', 'w', encoding='utf-8') as f:
        f.writelines([u + '\n' for u in outs])

    references = [[u] for u in tars]
    tmp = copy.deepcopy(references)
    bleu = count_score(outs, tmp, config)
    print(bleu)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
