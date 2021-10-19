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


class SimpleDecoder(torch.nn.Module):

    def __init__(self, configure):
        super(SimpleDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["num_words"],
                                            embedding_dim=configure["embedding_dim"])

        self.gru = torch.nn.GRU(input_size=configure["embedding_dim"],
                                hidden_size=configure["hidden_size"],
                                num_layers=configure["num_layers"],
                                bidirectional=False,
                                batch_first=True)

        self.fc = torch.nn.Linear(configure["hidden_size"], configure["num_words"])

    def forward(self, input, hidden):
        # Embedding
        embedding = self.embedding(input)

        # Call the GRU
        out, hidden = self.gru(embedding, hidden)

        out = self.fc(out.view(out.size(0), -1))

        return out, hidden


class AttentionDecoder(torch.nn.Module):

    def __init__(self, configure, device):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure
        self.device = device
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["num_words"],
                                            embedding_dim=configure["embedding_dim"])

        self.gru = torch.nn.LSTM(input_size=configure["embedding_dim"] + configure["hidden_size"],
                                 hidden_size=configure["hidden_size"],
                                 num_layers=configure["num_layers"],
                                 bidirectional=False,
                                 batch_first=True)

        self.att = Attention(configure["hidden_size"])

        self.fc = torch.nn.Linear(configure["hidden_size"], configure["num_words"])

        self.p = torch.nn.Linear(2 * configure["embedding_dim"] + configure["hidden_size"], 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden, encoder_output, z, content, coverage):
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
        attn_value = torch.zeros([attn.size(0), self.configure["num_words"]]).to(self.device)
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

    def forward(self, batch_src, batch_tar=None):
        encoder_out, encoder_hidden = self.encoder(batch_src[0])

        decoder_input = torch.tensor([101] * len(batch_src[0]), dtype=torch.long, device=self.config.device).view(len(batch_src[0]), -1)
        decoder_hidden = encoder_hidden
        z = torch.ones([len(batch_src[0]),1,self.config.hidden_size]).to(self.config.device)
        coverage = torch.zeros([self.config.batch_size,self.config.pad_size]).to(self.config.device)



def build(hidden_size, batch_size):
    x = import_module('config')
    bert_model = 'hfl/chinese-bert-wwm-ext'
    config = x.Config(batch_size, bert_model)
    encoder = SimpleEncoder(len(config.tokenizer.vocab), hidden_size, config.num_layers)
