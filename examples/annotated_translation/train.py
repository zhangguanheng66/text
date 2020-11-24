import numpy as np
import torch
import torchtext
from torch.autograd import Variable
from torchtext.experimental.vocab import load_vocab_from_file
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Generator, PositionalEncoding, AnnotatedTransformer
from torchtext.experimental.vocab import build_vocab_from_iterator

##############
# Prepare data
##############
raw_train_iter, raw_valid_iter, raw_test_iter = torchtext.experimental.datasets.raw.IWSLT()


def build_vocab(lines, tokenizer):
    vocab = build_vocab_from_iterator(list(tokenizer(line) for line in lines))
    vocab.insert_token('<pad>', 1)
    vocab.insert_token('<bos>', 2)
    vocab.insert_token('<eos>', 3)
    return vocab


de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')
de_vocab = build_vocab([lines[0] for lines in raw_train_iter], de_tokenizer)
en_vocab = build_vocab([lines[1] for lines in raw_train_iter], en_tokenizer)


def data_process(raw_iter):
    data_ = []
    for (raw_de, raw_en) in raw_iter:
        data_.append((de_vocab(de_tokenizer(raw_de)), en_vocab(en_tokenizer(raw_en))))
    return data_

train_data = data_process(raw_train_iter)
val_data = data_process(raw_valid_iter)
test_data = data_process(raw_test_iter)
print(len(train_data))
print(len(val_data))
print(len(test_data))
print(train_data[100])
print(val_data[100])
print(test_data[100])
