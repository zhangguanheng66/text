# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model


def pad_squad_data(batch):
    # Find max length of the mini-batch
    seq_list = []
    ans_pos_list = []
    seq_len = []

    for item in batch:
        seq_list.append(torch.cat((item['context'], item['question'])))
        seq_len.append(seq_list[-1].size(0))
        ans_pos_list.append(item['ans_pos'])

    max_l = max(seq_len)
    padded_tensors = torch.stack([torch.cat((txt,
                                  torch.tensor([pad_id] * (max_l - len(txt))).long()))
                                  for txt in seq_list]).t().contiguous()
    return padded_tensors.to(device), torch.stack(ans_pos_list).to(device)


###############################################################################
# Evaluating code
###############################################################################

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    batch_size = args.batch_size
    dataloader = DataLoader(data_source, batch_size=batch_size, collate_fn=pad_squad_data)

    with torch.no_grad():
        for idx, (seq_input, ans_pos) in enumerate(dataloader):
            start_pos, end_pos = model(seq_input)

            target_start_pos, target_end_pos = ans_pos.split(1, dim=-1)
            target_start_pos = target_start_pos.squeeze(-1)
            target_end_pos = target_end_pos.squeeze(-1)

            loss = (criterion(start_pos, target_start_pos) + criterion(end_pos, target_end_pos)) / 2
            total_loss += loss.item()

    return total_loss / len(data_source)


###############################################################################
# Training code
###############################################################################

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_squad_data)

    for idx, (seq_input, ans_pos) in enumerate(dataloader):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        start_pos, end_pos = model(seq_input)

        target_start_pos, target_end_pos = ans_pos.split(1, dim=-1)
        target_start_pos = target_start_pos.squeeze(-1)
        target_end_pos = target_end_pos.squeeze(-1)

        loss = (criterion(start_pos, target_start_pos) + criterion(end_pos, target_end_pos)) / 2
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, idx,
                                                      len(train_dataset),
                                                      scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / args.log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# At any point you can hit Ctrl + C to break out of training early.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    #parser.add_argument('--emsize', type=int, default=200,
    #                    help='size of word embeddings')
    #parser.add_argument('--nhid', type=int, default=200,
    #                    help='number of hidden units per layer')
    #parser.add_argument('--nlayers', type=int, default=2,
    #                    help='number of layers')
    #parser.add_argument('--lr', type=float, default=20,
    #                    help='initial learning rate')
    #parser.add_argument('--clip', type=float, default=0.25,
    #                    help='gradient clipping')
    #parser.add_argument('--epochs', type=int, default=40,
    #                    help='upper epoch limit')
    #parser.add_argument('--batch_size', type=int, default=20, metavar='N',
    #                    help='batch size')
    #parser.add_argument('--dropout', type=float, default=0.2,
    #                    help='dropout applied to layers (0 = no dropout)')
    #parser.add_argument('--tied', action='store_true',
    #                    help='tie the word embedding and softmax weights')
    #parser.add_argument('--seed', type=int, default=1111,
    #                    help='random seed')
    #parser.add_argument('--cuda', action='store_true',
    #                    help='use CUDA')
    #parser.add_argument('--log-interval', type=int, default=200, metavar='N',
    #                    help='report interval')
    #parser.add_argument('--save', type=str, default='model.pt',
    #                    help='path to save the final model')
    #
    #parser.add_argument('--nhead', type=int, default=2,
    #                    help='the number of heads in the encoder/decoder of the transformer model')

    # For test
    parser.add_argument('--emsize', type=int, default=32,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='qa_model.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--bert-model', type=str,
                        help='path to save the pretrained bert')

    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')

    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

#    import torchtext
#    from torchtext.experimental.datasets import WikiText103 as WikiData
#    from torchtext.experimental.datasets import WikiText2 as WikiData
    import torchtext
    from data import SQuAD
    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, dev_dataset = SQuAD()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)
    pad_id = vocab.stoi['<pad>']
    train_dataset, dev_dataset = SQuAD(vocab=vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    eval_batch_size = 16
#    train_data = batchify(train_dataset.data, args.batch_size)
#    dev_data = batchify(dev_dataset.data, eval_batch_size)




#    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_squad_data)
#
#for idx, (seq, ans_pos) in enumerate(dataloader):
#    print(idx, seq.size(), ans_pos.size(), ans_pos)




#    test_data = batchify(test_dataset.data, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

#    ntokens = len(train_dataset.get_vocab())
    pretrained_bert = torch.load(args.bert_model)
    model = model.QuestionAnswerTask(pretrained_bert).to(device)

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.25)
    best_val_loss = None

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(dev_dataset)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            scheduler.step()

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.

    # Run on test data.
    test_loss = evaluate(dev_dataset)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    with open('fine_tuning_qa_model.pt', 'wb') as f:
        torch.save(model, f)
#python qa_task.py --bert-model squad_vocab_pretrained_bert.pt --epochs 8 --save-vocab squad_vocab.pt
