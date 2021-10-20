import pickle

import torch
from utils.dataset import build_dataset
from tqdm import tqdm
import copy
from utils.bleu_eval import count_score
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from model import build
from utils.generator import generate_paragraph_data


def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(8, True)

    save_file_best = torch.load('./cache/best_save.data', map_location={'cuda:0': 'cuda:0'})

    model.load_state_dict(save_file_best['para'])
    model.eval()

    print('valid dataset:')
    generate_paragraph_data('../../data/test', '../../data/paragraph-level/data/test', config.tokenizer, mt5=True)

    test_data = build_dataset(config, '../../data/paragraph-level/data/test/src_ids.pkl',
                              '../../data/paragraph-level/data/test/src_masks.pkl',
                              '../../data/paragraph-level/data/test/tar_ids.pkl',
                              '../../data/paragraph-level/data/test/tar_masks.pkl',
                              '../../data/paragraph-level/data/test/tar_txts.pkl')

    max_len = 64
    outs = []
    tars = []
    for token_ids_src, _, seq_len_src, mask_src, token_ids_tar, __, seq_len_tar, mask_tar, tar_txt in tqdm(test_data):
        src = ''.join(config.tokenizer.convert_ids_to_tokens(token_ids_src))
        src = src.replace('▁', '').replace('</s>', '').replace('<pad>', '')
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
            src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
            src_masks = [[float(i != -100) for i in ii] for ii in src_ids]
            seq_len_srcs = len(src_ids[0]) - np.sum(src_ids == -100, axis=1)
            for ids, seq_len_src, masks in zip(src_ids, seq_len_srcs, src_masks):
                datas.append((ids, seq_len_src, masks))

            x_src = torch.LongTensor([_[0] for _ in datas]).to(config.device)
            seq_len = torch.LongTensor([_[1] for _ in datas])
            masks = torch.LongTensor([_[2] for _ in datas]).to(config.device)
            batch_src = (x_src, seq_len, masks)
            with torch.no_grad():
                outputs = model.generate(input_ids=batch_src[0],attention_mask=batch_src[2],do_sample=False,max_length=config.pad_size*2)
                sentences = [config.tokenizer.decode(u, skip_special_tokens=True) for u in outputs]
                sentences = [u.replace('▁','').replace('</s>','').replace('<pad>','') for u in sentences]

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
    with open('./result/paragraph_out.pkl','wb') as f:
        pickle.dump(outs,f)
    with open('./result/paragraph_ref.pkl', 'wb') as f:
        pickle.dump(tars, f)
    references = [[u] for u in tars]
    tmp = copy.deepcopy(references)
    bleu = count_score(outs, tmp, config)
    print(bleu)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    with open('./result/paragraph_out.pkl', 'rb') as f:
        outs = pickle.load(f)
    with open('./result/paragraph_ref.pkl', 'rb') as f:
        refs = pickle.load(f)
    outs = [' '.join(u) for u in outs]
    refs = [' '.join(u) for u in refs]

    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(outs, refs, avg=True)
    from pprint import pprint
    pprint(scores)