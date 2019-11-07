import torch
import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import read_text_iterator, create_data_from_iterator

URLS = {
    'Multi30k': ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
                 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
                 'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz']
}


def _setup_datasets(url, languages=('de', 'en'),
                    tokenizer=(get_tokenizer("spacy", language='de'),
                               get_tokenizer("spacy", language='en')),
                    root='.data', vocab=(None, None),
                    removed_tokens=['<unk>']):
    src_vocab, tgt_vocab = vocab
    src_language, tgt_language = languages
    src_tokenizer, tgt_tokenizer = tokenizer
    dataset_tar = download_from_url(url, root=root)
    extracted_files = extract_archive(dataset_tar)

    src_path = None
    tgt_path = None
    for fname in extracted_files:
        src_path = fname if src_language in fname else src_path
        tgt_path = fname if tgt_language in fname else tgt_path
    if not (src_path and tgt_path):
        raise TypeError("Source files are not found for the input languages")

    if src_vocab is None:
        logging.info('Building src Vocab based on train data')
        src_vocab = build_vocab_from_iterator(read_text_iterator(src_path, src_tokenizer))
    else:
        if not isinstance(src_vocab, Vocab):
            raise TypeError("Passed src vocabulary is not of type Vocab")
    logging.info('src Vocab has {} entries'.format(len(src_vocab)))

    if tgt_vocab is None:
        logging.info('Building tgt Vocab based on train data')
        tgt_vocab = build_vocab_from_iterator(read_text_iterator(tgt_path, tgt_tokenizer))
    else:
        if not isinstance(tgt_vocab, Vocab):
            raise TypeError("Passed tgt vocabulary is not of type Vocab")
    logging.info('tgt Vocab has {} entries'.format(len(tgt_vocab)))

    logging.info('Creating data')
    src_data = list(create_data_from_iterator(src_vocab,
                                              read_text_iterator(src_path, src_tokenizer),
                                              removed_tokens))
    tgt_data = list(create_data_from_iterator(tgt_vocab,
                                              read_text_iterator(tgt_path, tgt_tokenizer),
                                              removed_tokens))
    return TranslationDataset(list(zip(src_data, tgt_data)), (src_vocab, tgt_vocab))


class TranslationDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:
             - WikiText2
             - WikiText103
             - PennTreebank
    """

    def __init__(self, data, vocab):
        """Initiate language modeling dataset.
        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.Tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.
        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> data = torch.Tensor([token_id_1, token_id_2,
                                     token_id_3, token_id_1]).long()
            >>> vocab = build_vocab_from_iterator([['language', 'modeling']])
            >>> dataset = LanguageModelingDataset(data, vocab)
        """

        super(TranslationDataset, self).__init__()
        self._data = data
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_vocab(self):
        return self._vocab


def Multi30k(*args, **kwargs):
    """ Defines IMDB datasets.
        The labels includes:
            - 0 : Negative
            - 1 : Positive
    Create sentiment analysis dataset: IMDB
    Separately returns the training and test dataset
    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')
    Examples:
        >>> from torchtext.datasets import IMDB
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset = IMDB(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
    """

    train_dataset = _setup_datasets(*((URLS['Multi30k'][0],) + args), **kwargs)
    valid_dataset = _setup_datasets(*((URLS['Multi30k'][1],) + args), **kwargs)
    test_dataset = _setup_datasets(*((URLS['Multi30k'][2],) + args), **kwargs)
    return (train_dataset, valid_dataset, test_dataset)
