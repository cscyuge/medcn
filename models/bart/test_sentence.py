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
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(64, True)

    save_file_best = torch.load('./cache/best_save.data', map_location={'cuda:1': 'cuda:0'})

    model.load_state_dict(save_file_best['para'])
    model.eval()
    results = []
    references = []

    for i, (batch_src, batch_tar, batch_tar_txt) in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model.generate(batch_src[0])
            results += [config.tokenizer.decode(u, skip_special_tokens=True) for u in outputs]
            references += batch_tar_txt

    with open('./result/sentence_out.txt', 'w', encoding='utf-8') as f:
        f.writelines([u + '\n' for u in results])
    with open('./result/sentence_out.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('./result/sentence_ref.pkl', 'wb') as f:
        pickle.dump(references, f)

    refs = references[:]
    hyps = results[:]
    refs = [[u] for u in refs]
    bleu = count_score(hyps, refs, config)
    print(bleu)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    with open('./result/sentence_out.pkl', 'rb') as f:
        outs = pickle.load(f)
    with open('./result/sentence_ref.pkl', 'rb') as f:
        refs = pickle.load(f)
    outs = [' '.join(u) for u in outs]
    refs = [' '.join(u) for u in refs]

    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(outs, refs, avg=True)
    from pprint import pprint
    pprint(scores)
