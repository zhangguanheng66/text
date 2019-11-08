import torch
import logging
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import read_text_iterator, create_data_from_iterator
import os
import io
import codecs
import xml.etree.ElementTree as ET

URLS = {
    'Multi30k': ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
                 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
                 'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/'
                 'mmt_task1_test2016.tar.gz'],
    'WMT14': 'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'
}


def _clean_xml_file(f_xml):
    f_txt = os.path.splitext(f_xml)[0]
    with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
        root = ET.parse(f_xml).getroot()[0]
        for doc in root.findall('doc'):
            for e in doc.findall('seg'):
                fd_txt.write(e.text.strip() + '\n')


def _clean_tags_file(f_orig):
    xml_tags = ['<url', '<keywords', '<talkid', '<description',
                '<reviewer', '<translator', '<title', '<speaker']
    f_txt = f_orig.replace('.tags', '')
    with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
            io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
        for l in fd_orig:
            if not any(tag in l for tag in xml_tags):
                fd_txt.write(l.strip() + '\n')


def _setup_datasets(url, filenames=('de', 'en'),
                    tokenizer=(get_tokenizer("spacy", language='de'),
                               get_tokenizer("spacy", language='en')),
                    root='.data', vocab=(None, None),
                    removed_tokens=['<unk>']):
    src_vocab, tgt_vocab = vocab
    src_filename, tgt_filename = filenames
    src_tokenizer, tgt_tokenizer = tokenizer
    dataset_tar = download_from_url(url, root=root)
    extracted_files = extract_archive(dataset_tar)

    # Clean the xml and tag file in the archives
    file_archives = []
    for fname in extracted_files:
        if 'xml' in fname:
            _clean_xml_file(fname)
            file_archives.append(os.path.splitext(fname)[0])
        elif "tags" in fname:
            _clean_tags_file(fname)
            file_archives.append(fname.replace('.tags', ''))
        else:
            file_archives.append(fname)

    src_path = None
    tgt_path = None
    for fname in file_archives:
        src_path = fname if src_filename in fname else src_path
        tgt_path = fname if tgt_filename in fname else tgt_path
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
    src_data_iter = create_data_from_iterator(src_vocab,
                                              read_text_iterator(src_path, src_tokenizer),
                                              removed_tokens)
    src_data = [torch.tensor(t).long() for t in src_data_iter]

    tgt_data_iter = create_data_from_iterator(tgt_vocab,
                                              read_text_iterator(tgt_path, tgt_tokenizer),
                                              removed_tokens)
    tgt_data = [torch.tensor(t).long() for t in tgt_data_iter]

    return TranslationDataset(list(zip(src_data, tgt_data)), (src_vocab, tgt_vocab))


class TranslationDataset(torch.utils.data.Dataset):
    """Defines a dataset for translation.
       Currently, we only support the following datasets:
             - Multi30k
             - WMT14
    """

    def __init__(self, data, vocab):
        """Initiate translation dataset.

        Arguments:
            data: a tuple of source and target tensors, which include token ids
                numericalizing the string tokens.
                [(src_tensor0, tgt_tensor0), (src_tensor1, tgt_tensor1)]
            vocab: source and target Vocabulary object used for dataset.
                (src_vocab, tgt_vocab)

        Examples:
            >>> from torchtext.vocab import build_vocab_from_iterator
            >>> src_data = torch.Tensor([token_id_s1, token_id_s2,
                                         token_id_s3, token_id_s1]).long()
            >>> tgt_data = torch.Tensor([token_id_t1, token_id_t2,
                                         token_id_t3, token_id_t1]).long()
            >>> src_vocab = build_vocab_from_iterator([['Übersetzungsdatensatz']])
            >>> tgt_vocab = build_vocab_from_iterator([['translation', 'dataset']])
            >>> dataset = TranslationDataset([(src_data, tgt_data)],
                                              (src_vocab, tgt_vocab))
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


def Multi30k(tokenizer=(get_tokenizer("spacy", language='de'),
                        get_tokenizer("spacy", language='en')),
             root='.data', vocab=(None, None),
             removed_tokens=['<unk>']):

    """ Define translation datasets: Multi30k
        Separately returns train/valid/test datasets

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.de', 'train.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('val.de', 'val.en')
        test_filenames: the source and target filenames for test.
            Default: ('test2016.de', 'test2016.en')
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            Default: (torchtext.data.utils.get_tokenizer("spacy", language='de'),
                      torchtext.data.utils.get_tokenizer("spacy", language='en'))
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')

    Examples:
        >>> from torchtext.datasets import Multi30k
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = (get_tokenizer("spacy", language='de'),
                         get_tokenizer("basic_english"))
        >>> train_dataset, valid_dataset, test_dataset = Multi30k(tokenizer=tokenizer)
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """

    train_dataset = _setup_datasets(URLS['Multi30k'][0], ('train.de', 'train.en'),
                                    tokenizer, root, vocab, removed_tokens)
    src_vocab, tgt_vocab = train_dataset.get_vocab()
    valid_dataset = _setup_datasets(URLS['Multi30k'][1], ('val.de', 'val.en'), tokenizer,
                                    root, (src_vocab, tgt_vocab), removed_tokens)
    test_dataset = _setup_datasets(URLS['Multi30k'][2], ('test2016.de', 'test2016.en'),
                                   tokenizer, root,
                                   (src_vocab, tgt_vocab), removed_tokens)
    return (train_dataset, valid_dataset, test_dataset)


def IWSLT(languages='de-en',
          train_filenames=('train.de-en.de',
                           'train.de-en.en'),
          valid_filenames=('IWSLT16.TED.tst2013.de-en.de',
                           'IWSLT16.TED.tst2013.de-en.en'),
          test_filenames=('IWSLT16.TED.tst2014.de-en.de',
                          'IWSLT16.TED.tst2014.de-en.en'),
          tokenizer=(get_tokenizer("spacy", language='de'),
                     get_tokenizer("spacy", language='en')),
          root='.data', vocab=(None, None),
          removed_tokens=['<unk>']):

    """ Define translation datasets: IWSLT
        Separately returns train/valid/test datasets
        The available datasets include:

    Arguments:
        languages: the source and target languages for the datasets.
            Default: 'de-en' for source-target languages.
        train_filenames: the source and target filenames for training.
            Default: ('train.de-en.de', 'train.de-en.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('IWSLT16.TED.tst2013.de-en.de', 'IWSLT16.TED.tst2013.de-en.en')
        test_filenames: the source and target filenames for test.
            Default: ('IWSLT16.TED.tst2014.de-en.de', 'IWSLT16.TED.tst2014.de-en.en')
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            Default: (torchtext.data.utils.get_tokenizer("spacy", language='de'),
                      torchtext.data.utils.get_tokenizer("spacy", language='en'))
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')

    Examples:
        >>> from torchtext.datasets import IWSLT
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = IWSLT(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """
    src_language, tgt_language = languages.split('-')
    base_url = 'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
    url = base_url.format(src_language, tgt_language, languages)

    train_dataset = _setup_datasets(url, train_filenames,
                                    tokenizer, root, vocab, removed_tokens)
    src_vocab, tgt_vocab = train_dataset.get_vocab()
    valid_dataset = _setup_datasets(url, valid_filenames, tokenizer, root,
                                    (src_vocab, tgt_vocab), removed_tokens)
    test_dataset = _setup_datasets(url, test_filenames, tokenizer, root,
                                   (src_vocab, tgt_vocab), removed_tokens)
    return (train_dataset, valid_dataset, test_dataset)


def WMT14(train_filenames=('train.tok.clean.bpe.32000.de',
                           'train.tok.clean.bpe.32000.en'),
          valid_filenames=('newstest2013.tok.bpe.32000.de',
                           'newstest2013.tok.bpe.32000.en'),
          test_filenames=('newstest2014.tok.bpe.32000.de',
                          'newstest2014.tok.bpe.32000.en'),
          tokenizer=(get_tokenizer("spacy", language='de'),
                     get_tokenizer("spacy", language='en')),
          root='.data', vocab=(None, None),
          removed_tokens=['<unk>']):

    """ Define translation datasets: WMT14
        Separately returns train/valid/test datasets
        The available datasets include:
            newstest2016.en
            newstest2016.de
            newstest2015.en
            newstest2015.de
            newstest2014.en
            newstest2014.de
            newstest2013.en
            newstest2013.de
            newstest2012.en
            newstest2012.de
            newstest2011.tok.de
            newstest2011.en
            newstest2011.de
            newstest2010.tok.de
            newstest2010.en
            newstest2010.de
            newstest2009.tok.de
            newstest2009.en
            newstest2009.de
            newstest2016.tok.de
            newstest2015.tok.de
            newstest2014.tok.de
            newstest2013.tok.de
            newstest2012.tok.de
            newstest2010.tok.en
            newstest2009.tok.en
            newstest2015.tok.en
            newstest2014.tok.en
            newstest2013.tok.en
            newstest2012.tok.en
            newstest2011.tok.en
            newstest2016.tok.en
            newstest2009.tok.bpe.32000.en
            newstest2011.tok.bpe.32000.en
            newstest2010.tok.bpe.32000.en
            newstest2013.tok.bpe.32000.en
            newstest2012.tok.bpe.32000.en
            newstest2015.tok.bpe.32000.en
            newstest2014.tok.bpe.32000.en
            newstest2016.tok.bpe.32000.en
            train.tok.clean.bpe.32000.en
            newstest2009.tok.bpe.32000.de
            newstest2010.tok.bpe.32000.de
            newstest2011.tok.bpe.32000.de
            newstest2013.tok.bpe.32000.de
            newstest2012.tok.bpe.32000.de
            newstest2014.tok.bpe.32000.de
            newstest2016.tok.bpe.32000.de
            newstest2015.tok.bpe.32000.de
            train.tok.clean.bpe.32000.de

    Arguments:
        train_filenames: the source and target filenames for training.
            Default: ('train.tok.clean.bpe.32000.de', 'train.tok.clean.bpe.32000.en')
        valid_filenames: the source and target filenames for valid.
            Default: ('newstest2013.tok.bpe.32000.de', 'newstest2013.tok.bpe.32000.en')
        test_filenames: the source and target filenames for test.
            Default: ('newstest2014.tok.bpe.32000.de', 'newstest2014.tok.bpe.32000.en')
        tokenizer: the tokenizer used to preprocess source and target raw text data.
            Default: (torchtext.data.utils.get_tokenizer("spacy", language='de'),
                      torchtext.data.utils.get_tokenizer("spacy", language='en'))
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Source and target Vocabulary objects used for dataset. If None, it
            will generate a new vocabulary based on the train data set.
        removed_tokens: removed tokens from output dataset (Default: '<unk>')

    Examples:
        >>> from torchtext.datasets import WMT14
        >>> from torchtext.data.utils import get_tokenizer
        >>> src_tokenizer = get_tokenizer("spacy", language='de')
        >>> tgt_tokenizer = get_tokenizer("basic_english")
        >>> train_dataset, valid_dataset, test_dataset = WMT14(tokenizer=(src_tokenizer,
                                                                          tgt_tokenizer))
        >>> src_vocab, tgt_vocab = train_dataset.get_vocab()
        >>> src_data, tgt_data = train_dataset[10]
    """
    train_dataset = _setup_datasets(URLS['WMT14'], train_filenames,
                                    tokenizer, root, vocab, removed_tokens)
    src_vocab, tgt_vocab = train_dataset.get_vocab()
    valid_dataset = _setup_datasets(URLS['WMT14'], valid_filenames, tokenizer,
                                    root, (src_vocab, tgt_vocab), removed_tokens)
    test_dataset = _setup_datasets(URLS['WMT14'], test_filenames, tokenizer,
                                   root, (src_vocab, tgt_vocab), removed_tokens)
    return (train_dataset, valid_dataset, test_dataset)
