import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.experimental.vocab import build_vocab_from_iterator
import time
from model import AnnotatedTransformer
from torch import nn


###################################
# Prepare data
###################################
def build_vocab(lines, tokenizer):
    vocab = build_vocab_from_iterator(list(tokenizer(line) for line in lines))
    vocab.insert_token('<pad>', 1)
    vocab.insert_token('<bos>', 2)
    vocab.insert_token('<eos>', 3)
    return vocab


de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')
raw_train_iter, = torchtext.experimental.datasets.raw.IWSLT(data_select='train')
de_vocab = build_vocab([lines[0] for lines in raw_train_iter], de_tokenizer)
raw_train_iter, = torchtext.experimental.datasets.raw.IWSLT(data_select='train')
en_vocab = build_vocab([lines[1] for lines in raw_train_iter], en_tokenizer)


def data_process(raw_iter):
    data_ = []
    for (raw_de, raw_en) in raw_iter:
        data_.append((torch.tensor(de_vocab(de_tokenizer(raw_de)), dtype=torch.long),
                     torch.tensor(en_vocab(en_tokenizer(raw_en)), dtype=torch.long)))
    return data_


raw_train_iter, raw_valid_iter, raw_test_iter = torchtext.experimental.datasets.raw.IWSLT()
train_data = data_process(raw_train_iter)
val_data = data_process(raw_valid_iter)
test_data = data_process(raw_test_iter)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']
MAX_LEN = 64


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_item = torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0)
        if de_item.size(0) > MAX_LEN:
            de_item = de_item[:MAX_LEN]
        en_item = torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0)
        if en_item.size(0) > MAX_LEN:
            en_item = en_item[:MAX_LEN]
        de_batch.append(de_item)
        en_batch.append(en_item)
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)

# NLAYERS = 12
# EMB_DIM = 512
# HID_DIM = 2048
# NHEADS = 8

NLAYERS = 2
EMB_DIM = 128
HID_DIM = 512
NHEADS = 16
LR = 0.005  # learning rate

model = AnnotatedTransformer(INPUT_DIM, OUTPUT_DIM, N=NLAYERS,
                             d_model=EMB_DIM, d_ff=HID_DIM, h=NHEADS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.75)
mask_func = model.transformer.generate_square_subsequent_mask
# print(INPUT_DIM, OUTPUT_DIM)


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, tgt) in enumerate(iterator):

        optimizer.zero_grad()
        src, tgt = src.to(device), tgt.to(device)
        # print("src.size(), tgt.size(): ", src.size(), tgt.size())
        src_mask = mask_func(src.size(0)).to(device)
        tgt_mask = mask_func(tgt.size(0)).to(device)
        output = model(src, tgt, src_mask, tgt_mask)
        # print("output.view(-1, OUTPUT_DIM), tgt.view(-1): ", output.view(-1, OUTPUT_DIM).size(), tgt.view(-1).size())
        # break
        loss = criterion(output[1:].view(-1, OUTPUT_DIM), tgt[1:].view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, tgt) in enumerate(iterator):
            src, tgt = src.to(device), tgt.to(device)
            src_mask = mask_func(src.size(0)).to(device)
            tgt_mask = mask_func(tgt.size(0)).to(device)
            output = model(src, tgt, src_mask, tgt_mask)
            loss = criterion(output[1:].view(-1, OUTPUT_DIM), tgt[1:].view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {scheduler.get_lr()[0]:7.3f}')
    print(f'\tTrain Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}')
    scheduler.step()
test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test loss: {test_loss:7.3f} |')
