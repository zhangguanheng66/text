# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from model import MLMTask


def batchify(txt_data, bsz):

    # Cut the data to bptt and bsz
    _num = len(txt_data) // (bsz * args.bptt)
    txt_data = txt_data[:(_num * bsz * args.bptt)]
    # Divide the dataset into bsz parts.
    nbatch = txt_data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    txt_data = txt_data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    txt_data = txt_data.view(bsz, -1).t().contiguous()
    return txt_data.to(device)


###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
#    target = source[i+1:i+1+seq_len].view(-1)
#    return data, target
    return data

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    mask_id = train_dataset.vocab.stoi['<MASK>']
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data = get_batch(data_source, i)

            # Generate masks with args.mask_frac
            data_len = data.size(0)
            ones_num = int(data_len * args.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)].to(device)

            targets = torch.stack([data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
            data = data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)

            output = model(data)
            output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    mask_id = train_dataset.vocab.stoi['<MASK>']
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

        data = get_batch(train_data, i)

        # Generate masks with args.mask_frac
        data_len = data.size(0)
        ones_num = int(data_len * args.mask_frac)
        zeros_num = data_len - ones_num
        lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
        lm_mask = lm_mask[torch.randperm(data_len)].to(device)

        targets = torch.stack([data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
        data = data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        output = model(data)
        output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
#        print('targets.size() ', targets.size())
#        print('output.size() ', output.size())
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, scheduler.get_last_lr()[0],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)



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
    #parser.add_argument('--bptt', type=int, default=35,
    #                    help='sequence length')
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
    #parser.add_argument('--onnx-export', type=str, default='',
    #                    help='path to export the final model in onnx format')
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
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
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
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--save-vocab', type=str, default='vocab.pt',
                        help='path to save the vocab')

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

    import torchtext
#    from torchtext.experimental.datasets import WikiText103 as WikiData
    from torchtext.experimental.datasets import WikiText2 as WikiData
    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, test_dataset, valid_dataset = WikiData()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)
    train_dataset, test_dataset, valid_dataset = WikiData(vocab=vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_batch_size = 10
    train_data = batchify(train_dataset.data, args.batch_size)
    val_data = batchify(valid_dataset.data, eval_batch_size)
    test_data = batchify(test_dataset.data, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(train_dataset.get_vocab())
#    print(ntokens)
    model = MLMTask(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.25)
    best_val_loss = None

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
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
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

# python main.py --seed 68868 --epochs 12 --emsize 256 --nhid 1024  --nlayers 16 --nhead 16 --save-vocab squad_vocab.pt
