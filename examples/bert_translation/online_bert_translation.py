import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
from tokenizers import BertWordPieceTokenizer

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

#This is a translator that goes from German to English
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")

#set all random seeds for reproducability
# SEED = 1234
#
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


#setup tokenizers for english and german
tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
def tokenize_en(text):
    return (tokenizer.encode(text).tokens[1:-1])[:98]

#define torchtext Fields
FIELD = Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True)

#load data
train_data = TranslationDataset(path="data/train1", exts = ('.from', '.to'), fields = (FIELD, FIELD))
valid_data = TranslationDataset(path="data/valid1", exts = ('.from', '.to'), fields = (FIELD, FIELD))
test_data = TranslationDataset(path="data/test", exts = ('.from', '.to'), fields = (FIELD, FIELD))

FIELD.build_vocab(train_data)
vocabLines = open("bert-base-uncased-vocab.txt", "rb").readlines()
for i in range(len(vocabLines)):
    vocabLines[i] = vocabLines[i].decode("utf-8").replace("\n", "")
    FIELD.vocab.stoi[vocabLines[i]] = i
    if i >= len(FIELD.vocab.itos):
        FIELD.vocab.itos.append(vocabLines[i])
    else:
        FIELD.vocab.itos[i] = vocabLines[i]


BATCH_SIZE = 1

#data iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device)

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, multi_head_attention_layer, positionwise_feedforward_layer, dropout, device):
        super(Encoder, self).__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(100, hid_dim)

        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, multi_head_attention_layer, positionwise_feedforward_layer, dropout, device) for _ in range(n_layers)]) # the main encoder layers

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        #src = (batch size, seq length)
        #src_mask = (batch size, seq length)

        batch_size = src.shape[0]
        seq_len = src.shape[1]

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) # create the position indexes as a range counting from 0 to seq_len and repeat it for each one in the batch
        #pos = (batch size, seq length)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)) # embed the tokens and the positions, scale the token embeddings, and sum the embeddings together
        #src = (batch size, seq length, hid dim)

        for layer in self.layers: # feed through each encoder layer
            src = layer(src, src_mask)

        return(src)

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, multi_head_attention_layer, positionwise_feedforward_layer, dropout, device):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = multi_head_attention_layer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward_layer = positionwise_feedforward_layer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src = (batch size, seq length, hid dim)
        #src_mask = (batch size, seq length)

        #self attention
        _src, _ = self.self_attention(src, src, src, mask=src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = (batch size, seq length, hid dim)

        #positionwise feedforward
        _src = self.positionwise_feedforward_layer(src)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = (batch size, seq len, hid dim)

        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert(hid_dim % n_heads == 0) # make sure we can cleanly divide the inputs into the heads

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim) # query
        self.fc_k = nn.Linear(hid_dim, hid_dim) # key
        self.fc_v = nn.Linear(hid_dim, hid_dim) # value
        self.fc_o = nn.Linear(hid_dim, hid_dim) # output

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, query, key, value, mask=None): # we take in the query, key and value seperately so we can reuse this layer for both self attention as well as encoder-decoder attention
        batch_size = query.shape[0]

        #query = (batch size, query length, hid dim)
        #key = (batch size, key length, hid dim)
        #value = (batch size, value length, hid dim)

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = (batch size, query length, hid dim)
        #K = (batch size, key length, hid dim)
        #V = (batch size, value length, hid dim)

        # split query, key and value into the heads
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = (batch size, n heads, query length, head dim)
        #K = (batch size, n heads, key length, head dim)
        #V = (batch size, n heads, value length, head dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = (batch size, n heads, seq length, seq length)

        if mask is not None: #apply the mask
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        #attention = (batch size, n heads, query len, key len)

        x = torch.matmul(self.dropout(attention), V)

        #x = (batch size, n heads, seq length, head dim)

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = (batch size, seq length, n heads, head dim)

        x = x.view(batch_size, -1, self.hid_dim) #combine heads

        #x = (batch size, seq len, hid dim)

        x = self.fc_o(x)
        return(x, attention)

class PositionWiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionWiseFeedforwardLayer, self).__init__()
        # contains two linear layers with a nonlinearity in between. The intermediate size (pf_dim) is usually much bigger
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = (batch size, seq length, hid dim)

        x = self.dropout(torch.relu(self.fc_1(x)))

        #x = (batch size, seq length, pf dim)

        x = self.fc_2(x)

        #x = (batch size, seq length, hid dim)
        return(x)

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, multi_head_attention_layer, positionwise_feedforward_layer, dropout, device):
        super(Decoder, self).__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(100, hid_dim)

        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, multi_head_attention_layer, positionwise_feedforward_layer, dropout, device)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = (batch size, trg len)
        #enc_src = (batch size, src length, hid dim)
        #trg mask = (batch size, trg len)
        #src mask = (batch size, src len)

        batch_size = trg.shape[0]
        trg_length = trg.shape[1]

        pos = torch.arange(0, trg_length).unsqueeze(0).repeat(batch_size, 1).to(self.device) # create the position indexes as a range counting from 0 to seq_len and repeat it for each one in the batch
        #pos = (batch size, trg length)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg = (batch size, trg length, hid dim)

        for layer in self.layers: # run through the decoder layers
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = (batch size, trg length, hid dim)
        #attention = (batch size, n heads, trg len, src len)

        output = self.fc_out(trg)
        #output = (batch size, trg len, output dim)

        return(output, attention)

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, multi_head_attention_layer, positionwise_feedforward_layer, dropout, device):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = multi_head_attention_layer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = multi_head_attention_layer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward_layer = positionwise_feedforward_layer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = (batch size, trg len, hid dim)
        #enc_src = (batch size, src len, hid dim)
        #trg_mask = (batch size, trg len)
        #src_mask = (batch size, src len)

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, mask=trg_mask)

        #dropout, residual layer and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = (batch size, trg len, hid dim)

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, mask=src_mask)

        #dropout, residual layer and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = (batch size, trg len, hid dim)

        #positionwise feedforward
        _trg = self.positionwise_feedforward_layer(trg)

        #dropout, residual layer and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = (batch size, trg len, hid dim)
        #attention = (batch size, n heads, trg len, src len)
        return(trg, attention)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        #src = (batch size, src len)

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = (batch size, 1, 1, src len)
        return(src_mask)

    def make_trg_mask(self, trg):
        #trg = (batch size, trg len)

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        #trg_mask = (batch size, 1, trg len, 1)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = (trg len, trg len)

        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = (batch size, 1, trg len, trg len)
        return(trg_mask)

    def forward(self, src, trg):
        #src = (batch size, src len)
        #trg = (batch size, trg len)

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = (batch size, 1, 1, src len)
        #trg_mask = (batch size, 1, trg len, trg len)

        enc_src = self.encoder(src, src_mask)

        #enc_src = (batch size, src len, hid dim)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = (batch size, trg len, output dim)
        #attention = (batch size, n heads, trg len, src len)

        return output, attention

#hyperparameters
INPUT_DIM = len(FIELD.vocab)
OUTPUT_DIM = len(FIELD.vocab)
HID_DIM = 252
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 250 # intermediate dimensions in the positionwise feedforward
DEC_PF_DIM = 250
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

PAD_IDX = FIELD.vocab.stoi[FIELD.pad_token]

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, EncoderLayer, MultiHeadAttentionLayer, PositionWiseFeedforwardLayer, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DecoderLayer, MultiHeadAttentionLayer, PositionWiseFeedforwardLayer, DEC_DROPOUT, device)
model = Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, device).to(device)

#helper functions
def count_parameters(model):
    return(sum(p.numel() for p in model.parameters() if p.requires_grad))
def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

print("The model has " + str(count_parameters(model)) + " trainable parameters")
model.apply(initialize_weights)

LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, threshold=0.01, cooldown=20)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, iterator, optimizer, criterion, clip, accumulation_batches=1):
    model.train() # activate training mode

    epoch_loss = 0
    loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1]) # take off the eos at the end of each since we don't want to feed it in

        #output = (batch size, trg len - 1, output dim)
        #trg = (batch size, trg len)

        # flatten for loss
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        #output = (batch size * (trg len - 1), output dim)
        #trg = (batch size * (trg len - 1))

        loss += criterion(output, trg)

        if (i + 1) % accumulation_batches == 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            loss = 0

    return(epoch_loss / len(iterator))

def evaluate(model, iterator, criterion):
    model.eval() # activate eval mode

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            #output = (batch size, trg len - 1, output dim)
            #trg = (batch size, trg len)

            #flatten for loss
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            #output = (batch size * (trg len - 1), output dim)
            #trg = (batch size * (trg len - 1))

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return(epoch_loss / len(iterator))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return(elapsed_mins, elapsed_secs)

N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')

# Main Train Loop
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, accumulation_batches=1)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "Seq2SeqModel.pt")
    lr = [ group['lr'] for group in optimizer.param_groups ]
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {lr}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t  Val Loss: {valid_loss:.3f} |   Val PPL: {math.exp(valid_loss):7.3f}')
    lr_scheduler.step(train_loss)

#model.load_state_dict(torch.load('Seq2SeqModel.pt')) # load model

# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval() # activate evaluation mode

    if isinstance(sentence, str): # tokenize if not tokenized
        tokens = tokenize_en(sentence)
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token] # add start and end tokens

    src_indexes = [src_field.vocab.stoi[token] for token in tokens] # get indicies for each token
    print("INPUT: " + str(src_indexes))
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device) # make tensor and add batch dimension

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad(): # run through encoder
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device) # add batch dim to target tensor

        trg_mask = model.make_trg_mask(trg_tensor) # make current decoder mask

        with torch.no_grad(): # run current output through decoder to get new output
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(dim=2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[index] for index in trg_indexes]

    return(trg_tokens[1:], attention)


def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15,25))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

# example
example_idx = 2
src = vars(train_data.examples[example_idx])["src"]
trg = vars(train_data.examples[example_idx])["trg"]
print(f'src = {src}')
print(f'trg = {trg}')
translation, attention = translate_sentence(src, FIELD, FIELD, model, device)
#display_attention(src, translation, attention)
print(f'predicted trg = {translation}')
example_idx = 1
src = vars(train_data.examples[example_idx])["src"]
trg = vars(train_data.examples[example_idx])["trg"]
print(f'src = {src}')
print(f'trg = {trg}')
translation, attention = translate_sentence(src, FIELD, FIELD, model, device)
#display_attention(src, translation, attention)
print(f'predicted trg = {translation}')



while True:gg
    customSent = input("Input A Sentence: ")
    translation, attention = translate_sentence(customSent, FIELD, FIELD, model, device)
    print("Translation: " + (" ".join(translation)).replace("<eos>", ""))
