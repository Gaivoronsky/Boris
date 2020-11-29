import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from collections import Counter
from typing import List, Tuple
import os

DEVICE = torch.device('cuda')


class Vocabulary:
    def __init__(self):
        self.index2word = list()
        self.word2index = dict()
        self.word2count = Counter()
        self.reset()

    def get_pad(self):
        return self.word2index["<pad>"]

    def get_sos(self):
        return self.word2index["<sos>"]

    def get_eos(self):
        return self.word2index["<eos>"]

    def get_unk(self):
        return self.word2index["<unk>"]

    def add_sentence(self, sentence):
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.word2count[word] += 1
            self.index2word.append(word)
        else:
            self.word2count[word] += 1

    def has_word(self, word) -> bool:
        return word in self.word2index

    def add_file(self, filename: str):
        with open(filename, "r", encoding="utf-8") as r:
            for line in r:
                for word in line.strip().split():
                    self.add_word(word)

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        return self.get_unk()

    def get_word(self, index):
        return self.index2word[index]

    def size(self):
        return len(self.index2word)

    def is_empty(self):
        empty_size = 4
        return self.size() <= empty_size

    def shrink(self, n):
        best_words = self.word2count.most_common(n)
        self.reset()
        for word, count in best_words:
            self.add_word(word)
            self.word2count[word] = count

    def reset(self):
        self.word2count = Counter()
        self.index2word = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.word2index = {word: index for index, word in enumerate(self.index2word)}

    def get_indices(self, sentence: str) -> List[int]:
        return [self.get_index(word) for word in sentence.strip().split()] + [self.get_eos()]

    def pad_indices(self, indices: List[int], max_length: int):
        return indices + [self.get_pad() for _ in range(max_length - len(indices))]


vocabulary = Vocabulary()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, n_layers=3, dropout=0.3, bidirectional=True):
        super(EncoderRNN, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_seqs, hidden=None):
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.rnn(embedded, hidden)

        if self.bidirectional:
            n = hidden[0].size(0)
            hidden = (torch.cat([hidden[0][0:n:2], hidden[0][1:n:2]], 2),
                      torch.cat([hidden[1][0:n:2], hidden[1][1:n:2]], 2))
        return outputs, hidden


class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        assert inputs.size(1) == self.hidden_size
        return self.sm(self.out(inputs))


class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, output_size, max_length, n_layers=3,
                 dropout=0.3, use_cuda=True, use_attention=True):
        super(DecoderRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.max_length = max_length
        self.use_attention = use_attention

        self.embedding = nn.Embedding(output_size, embedding_dim)

        if self.use_attention:
            self.attn = nn.Linear(embedding_dim, hidden_size)
            self.attn_sm = nn.Softmax(dim=1)
            self.attn_ctx = nn.Linear(hidden_size + hidden_size, hidden_size)
            self.attn_tanh = nn.Tanh()

        self.rnn = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=dropout)
        self.generator = Generator(hidden_size, output_size)

    def step(self, batch_input, hidden, encoder_output):
        # batch_input: B
        # hidden: (n_layers x B x N, n_layers x B x N)
        # encoder_output: L x B x N
        # output: 1 x B x N
        # embedded:  B x E
        # attn_weights: B x 1 x L
        # context: B x 1 x N
        # rnn_input: B x N

        embedded = self.embedding(batch_input)
        _, hidden = self.rnn(embedded.unsqueeze(0), hidden)
        if self.use_attention:
            mapped_encoder_outputs = self.attn(encoder_output)
            h_t = hidden[0][-1].unsqueeze(2)
            mapped_encoder_outputs = mapped_encoder_outputs.transpose(0, 1)
            attn_weights = self.attn_sm(torch.bmm(mapped_encoder_outputs, h_t)).transpose(1, 2)
            max_length = encoder_output.size(0)
            context = torch.bmm(attn_weights[:, :, :max_length], encoder_output.transpose(0, 1))
            output = self.attn_tanh(self.attn_ctx(torch.cat((context, h_t.transpose(1, 2)), 2)))
        return output, hidden

    def init_state(self, batch_size, sos_index):
        initial_input = Variable(torch.zeros((batch_size,)).type(torch.LongTensor), requires_grad=False)
        initial_input = torch.add(initial_input, sos_index)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input
        return initial_input

    def forward(self, current_input, hidden, length, encoder_output, gtruth=None):
        outputs = Variable(torch.zeros(length, current_input.size(0), self.output_size), requires_grad=False)
        outputs = outputs.cuda() if self.use_cuda else outputs

        for t in range(length):
            output, hidden = self.step(current_input, hidden, encoder_output)
            scores = self.generator.forward(output.squeeze(1))
            outputs[t] = scores
            if gtruth is None:
                top_indices = scores.topk(1, dim=1)[1].view(-1)
                current_input = top_indices
            else:
                current_input = gtruth[t]
        return outputs, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocabulary, embedding_dim=100,
                 rnn_size=100, encoder_n_layers=2,
                 decoder_n_layers=2, dropout=0.3,
                 max_length=50, use_cuda=True,
                 bidirectional=True, use_attention=True):
        super(Seq2Seq, self).__init__()

        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.output_size = vocabulary.size()
        self.rnn_size = rnn_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.encoder = EncoderRNN(self.output_size, embedding_dim, rnn_size, dropout=dropout,
                                  n_layers=encoder_n_layers, bidirectional=bidirectional)
        self.decoder = DecoderRNN(embedding_dim, rnn_size, self.output_size, dropout=dropout,
                                  max_length=max_length, n_layers=decoder_n_layers, use_cuda=use_cuda,
                                  use_attention=use_attention)

    def forward(self, variable, sos_index, gtruth=None):
        encoder_output, encoder_hidden = self.encoder.forward(variable)
        current_input = self.decoder.init_state(variable.size(1), sos_index)
        max_length = self.max_length
        if gtruth is not None:
            max_length = min(self.max_length, gtruth.size(0))
        decoder_output, _ = self.decoder.forward(current_input, encoder_hidden, max_length,
                                                 encoder_output, gtruth)

        return encoder_output, decoder_output


def index(vocabulary, line):
    line = line.strip().replace("\n", " ").replace("\xa0", " ").lower()
    tokens, _ = tokenizer.tokenize(line)
    return [vocabulary.get_index(token) for token in tokens]


def to_matrix(lines, vocabulary):
    indices = [index(vocabulary, line) for line in lines]
    max_len = max([len(line) for line in indices])
    matrix = np.zeros((len(indices), max_len))
    for i, line in enumerate(indices):
        matrix[i, :len(line)] = line
    return torch.cuda.LongTensor(matrix)


class Seq2SeqModelTrainer():
    def __init__(self, model, optimizer, vocabulary):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.NLLLoss()
        self.vocabulary = vocabulary

    def on_epoch_begin(self, is_train, name, batches_count):
        self.epoch_loss = 0
        self.is_train = is_train
        self.name = name
        self.batches_count = batches_count
        self.model.train(is_train)

    def on_epoch_end(self):
        return '{:>5s} Loss = {:.5f}'.format(self.name, self.epoch_loss / self.batches_count)

    def on_batch(self, batch):
        pivot_lines = batch['pivot_lines']
        positive_lines = batch['positive_lines']
        pivot_lines = [" ".join(context) for context in pivot_lines]

        input_matrix = to_matrix(pivot_lines, self.vocabulary).transpose(0, 1)
        output_matrix = to_matrix(positive_lines, self.vocabulary).transpose(0, 1)
        _, output = self.model.forward(input_matrix, self.vocabulary.get_sos(), output_matrix)
        loss = self.criterion(output.transpose(1, 2), output_matrix)
        self.epoch_loss += loss.item()

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

        return '{:>5s} Loss = {:.5f}'.format(self.name, loss.item())


model = Seq2Seq(vocabulary).to(DEVICE)
model.load_state_dict(torch.load('neural_network/model/model_new.pth'))

while True:
    user_text = input()
    input_matrix = to_matrix([user_text], vocabulary).transpose(0, 1)
    model.eval()
    _, output = model(input_matrix, model.vocabulary.get_sos())

    answer_idx = output.argmax(dim=2).cpu().numpy().ravel()
    print(' '.join(vocabulary.get_word(idx) for idx in answer_idx if idx != vocabulary.get_pad()))
