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

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'


class SimpleEncoder(torch.nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers, bidirectional=False):
        super(SimpleEncoder, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=hidden_size)

        self.gru = torch.nn.LSTM(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, vocab_size)


    def forward(self, input):

        # Embedding
        embedding = self.embedding(input)

        # Call the GRU
        out, hidden = self.gru(embedding)

        return out, hidden


class Attention(nn.Module):
    """ Simple Attention

    This Attention is learned from weight
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)

        # Declare the Attention Weight
        self.W = nn.Linear(dim, 1)

        # Declare the coverage feature
        self.coverage_feature = nn.Linear(1, dim)

    def forward(self, output, context, coverage):
        # declare the size
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # Expand the output to the num of timestep
        output_expand = output.expand(batch_size, input_size, hidden_size)

        # reshape to 2-dim
        output_expand = output_expand.reshape([-1, hidden_size])
        context = context.reshape([-1, hidden_size])

        # transfer the coverage to features
        coverage_feature = self.coverage_feature(coverage.reshape(-1, 1))

        # Learning the attention
        attn = self.W(output_expand + context + coverage_feature)
        attn = attn.reshape(-1, input_size)
        attn = F.softmax(attn, dim=1)

        # update the coverage
        coverage = coverage + attn

        context = context.reshape(batch_size, input_size, hidden_size)
        attn = attn.reshape(batch_size, -1, input_size)

        # get the value of a
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn, coverage


class AttentionDecoder(torch.nn.Module):

    def __init__(self, vocab, hidden_size, num_layers, device):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter
        self.device = device

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=vocab,
                                            embedding_dim=hidden_size)

        self.gru = torch.nn.LSTM(input_size=hidden_size*2,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=False,
                                 batch_first=True)

        self.att = Attention(hidden_size)

        self.fc = torch.nn.Linear(hidden_size, vocab)

        self.p = torch.nn.Linear(3 * hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, vocab, input, hidden, encoder_output, z, content, coverage):
        # Embedding
        embedding = self.embedding(input)
        # print(embedding.squeeze().size())

        combine = torch.cat([embedding, z], 2)
        # print(combine.squeeze().size())
        # Call the GRU
        out, hidden = self.gru(combine, hidden)

        # call the attention
        output, attn, coverage = self.att(output=out, context=encoder_output, coverage=coverage)

        index = content
        attn = attn.view(attn.size(0), -1)
        attn_value = torch.zeros([attn.size(0), vocab]).to(self.device)
        attn_value = attn_value.scatter_(1, index, attn)

        out = self.fc(output.view(output.size(0), -1))
        # print(torch.cat([embedding.squeeze(), combine.squeeze()], 1).size(), )
        p = self.sigmoid(self.p(torch.cat([embedding.squeeze(1), combine.squeeze(1)], 1)))
        # print(p)
        out = (1 - p) * out + p * attn_value
        # print(attn_value.size(), output.size())

        return out, hidden, output, attn, coverage


class PointerNet(nn.Module):
    def __init__(self, encoder, decoder, config):
        super(PointerNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()

    def flatten_parameters(self):
        self.encoder.gru.flatten_parameters()
        self.decoder.gru.flatten_parameters()

    def forward(self, batch_src, batch_tar=None):
        batch_size = batch_src[0].size(0)
        pad_size = batch_src[0].size(1)
        if batch_tar is not None:
            target = batch_tar[0]
        encoder_out, encoder_hidden = self.encoder(batch_src[0])
        decoder_input = torch.tensor([self.config.tokenizer.cls_token_id] * batch_size,
                                     dtype=torch.long, device=self.config.device).view(batch_size, -1)
        decoder_hidden = encoder_hidden
        z = torch.ones([batch_size, 1, self.config.hidden_size]).to(self.config.device)
        coverage = torch.zeros([batch_size, pad_size]).to(self.config.device)
        seq_loss = 0
        step_coverage_loss = 0
        sentence_symbols = []
        for i in range(pad_size):
            decoder_output, decoder_hidden, z, attn, coverage = self.decoder(len(self.config.tokenizer.vocab),decoder_input, decoder_hidden,
                                                                              encoder_out, z, batch_src[0], coverage)
            symbols = torch.max(decoder_output, 1)[1].cpu().tolist()
            sentence_symbols.append(symbols)

            if batch_tar is None or random.randint(1, 10) > 5:
                _, decoder_input = torch.max(decoder_output, 1)
                decoder_input = decoder_input.view(batch_size, -1)
            else:
                decoder_input = target[:, i].view(batch_size, -1)
            decoder_hidden = decoder_hidden
            if batch_tar is not None:
                step_coverage_loss = torch.sum(torch.min(attn.reshape(-1, 1), coverage.reshape(-1, 1)), 1)
                step_coverage_loss = torch.sum(step_coverage_loss)
                seq_loss += (self.criterion(decoder_output, target[:, i]))
                seq_loss += step_coverage_loss

        return seq_loss, step_coverage_loss, sentence_symbols


def build(hidden_size, batch_size, cuda=True):
    x = import_module('config')
    bert_model = 'hfl/chinese-bert-wwm-ext'
    config = x.Config(batch_size, bert_model)

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
    val_dataloader = build_iterator(val_data, config)
    test_dataloader = build_iterator(test_data, config)

    encoder = SimpleEncoder(len(config.tokenizer.vocab), hidden_size, config.num_layers)
    decoder = AttentionDecoder(len(config.tokenizer.vocab), hidden_size, config.num_layers, config.device)
    encoder = encoder.to(config.device)
    decoder = decoder.to(config.device)

    pointerNet = PointerNet(encoder, decoder, config)
    if cuda:
        pointerNet.cuda()
    pointerNet = pointerNet.to(config.device)
    optimizer = torch.optim.Adam(pointerNet.parameters(), lr=config.learning_rate)
    t_total = int(len(train_data) / config.batch_size * config.num_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config.warmup_proportion * t_total),
                                                num_training_steps=t_total)  # PyTorch scheduler


    return pointerNet, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, config


def decode_sentence(symbols, config):
    sentences = []
    for symbol_sen in symbols:
        words = config.tokenizer.convert_ids_to_tokens(symbol_sen)
        temp = ''
        for word in words:
            if word == '[SEP]':
                break
            if word[0] == '#':
                word = word[2:]
                temp += word
            else:
                temp += word
        sentences.append(temp)
    return sentences


def eval_set(model, dataloader, config):
    model.eval()
    results = []
    references = []

    for i, (batch_src, batch_tar, batch_tar_txt) in enumerate(dataloader):
        with torch.no_grad():
            seq_loss, step_coverage_loss, sentence_symbols = model(batch_src, None)
            symbols = np.array(sentence_symbols).T
            sentences = decode_sentence(symbols, config)
            results += sentences
            references += batch_tar_txt
    references = [[u] for u in references]
    tmp = copy.deepcopy(references)
    bleu = count_score(results, tmp, config)
    del tmp

    sentences = []
    for words in results:
        tmp = ''
        for word in words:
            tmp += word
        sentences.append(tmp)
    model.train()
    return sentences, bleu


def train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, config):
    #training steps
    max_bleu = -99999
    save_file = {}
    for e in range(config.num_epochs):
        model.train()
        for i, (batch_src, batch_tar, batch_tar_txt) in tqdm(enumerate(train_dataloader)):
            # words_1 = ''.join(config.tokenizer.convert_ids_to_tokens(batch_src[0][0]))
            # words_2 = ''.join(config.tokenizer.convert_ids_to_tokens(batch_tar[0][0]))
            seq_loss, step_coverage_loss, sentence_symbols = model(batch_src, batch_tar)

            seq_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                # print('sample:')
                # print(words_1)
                # print(words_2)
                print('train loss:%f' %(seq_loss.item()/len(batch_src[0][0])/len(batch_src[0])))


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
    seq2seq, optimizer,scheduler, train_dataloader, val_dataloader, test_dataloader, config = build(256, 128, True)
    bleu = train(seq2seq, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, config)
    print('finish')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()