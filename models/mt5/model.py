import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from importlib import import_module
from tqdm import tqdm
import copy
from utils.bleu_eval import count_score
from utils.dataset import build_dataset, build_iterator
from transformers import MT5ForConditionalGeneration

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'


def build(batch_size, cuda):

    x = import_module('config')
    pretrained_model = 'google/mt5-small'
    config = x.Config(batch_size, pretrained_model)
    test_config = x.Config(batch_size*4, pretrained_model)
    train_data = build_dataset(config, './data/train/src_ids.pkl', './data/train/src_masks.pkl',
                               './data/train/tar_ids.pkl',
                               './data/train/tar_masks.pkl', './data/train/tar_txts.pkl')
    test_data = build_dataset(config, './data/test/src_ids.pkl', './data/test/src_masks.pkl',
                              './data/test/tar_ids.pkl',
                              './data/test/tar_masks.pkl', './data/test/tar_txts.pkl')
    val_data = build_dataset(config, './data/valid/src_ids.pkl', './data/valid/src_masks.pkl',
                             './data/valid/tar_ids.pkl',
                             './data/valid/tar_masks.pkl', './data/valid/tar_txts.pkl')
    train_dataloader = build_iterator(train_data, config)
    val_dataloader = build_iterator(val_data, test_config)
    test_dataloader = build_iterator(test_data, test_config)

    model = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
    model = model.to(config.device)
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.learning_rate
    )

    return model, optimizer, train_dataloader, val_dataloader, test_dataloader, config


def eval_set(model, dataloader, config):
    model.eval()
    results = []
    references = []

    for i, (batch_src, batch_tar, batch_tar_txt) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch_src[0],attention_mask=batch_src[2],do_sample=False,max_length=config.pad_size*2)
            results += config.tokenizer.batch_decode(outputs,skip_special_tokens=True)
            references += batch_tar_txt

    references = [[u] for u in references]
    tmp = copy.deepcopy(references)
    bleu = count_score(results, tmp, config)
    del tmp

    sentences = []
    for words in results:
        tmp = ''
        for word in words:
            if word=='▁': continue
            tmp += word
        sentences.append(tmp)
    model.train()
    return sentences, bleu


def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config):
    #training steps
    max_bleu = -99999
    save_file = {}
    for e in range(config.num_epochs):
        model.train()
        for i, (batch_src, batch_tar, batch_tar_txt) in tqdm(enumerate(train_dataloader)):
            model_outputs = model(input_ids=batch_src[0], attention_mask=batch_src[2], labels=batch_tar[0])

            loss = model_outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 1000 == 0:
                print('train loss:%f' %loss.item())


        #validation steps
        if e >= 0:
            val_results, bleu = eval_set(model, val_dataloader, config)
            print(val_results[0:5])
            print('BLEU:%f' %(bleu))
            if bleu > max_bleu:
                max_bleu = bleu
                save_file['epoch'] = e + 1
                save_file['para'] = model.state_dict()
                save_file['best_bleu'] = bleu
                torch.save(save_file, './cache/best_save.data')
            if bleu < max_bleu - 0.6:
                print('Early Stop')
                break
            print(save_file['epoch'] - 1)


    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val BLEU:%f' %(save_file_best['best_bleu']))
    model.load_state_dict(save_file_best['para'])
    test_results, bleu = eval_set(model, test_dataloader, config)
    print('Test BLEU:%f' % (bleu))
    with open('./result/best_save_bert.out.txt', 'w', encoding="utf-8") as f:
        f.writelines([x + '\n' for x in test_results])
    return bleu

def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(8, True)
    bleu = train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config)
    # print('finish')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
