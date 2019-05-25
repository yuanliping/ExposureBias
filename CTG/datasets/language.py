from CTG.tokenizer import Tokenizer
from CTG.dictionary import Dictionary

import numpy as np

import torch
from torch.utils.data import Dataset


class LanguageTask():

    def __init__(self, args):
        self.dataset = {}
        print('Loading Data...')
        for split in args['splits']:
            self.dataset[split] = LaugnageDatasets('{}.{}.txt'.format(args['data'], split),
                                                   maxlen=args['maxlen'])
        print('Loading Done.')
        print('Building Dictionary...')
        self.dictionary = Dictionary()
        self.build_dictionary(self.dataset['train'].get_raw_dataset(), nwords=args['nwords'])
        print('Building Done.')
        print('Dictionary Size: {}'.format(len(self.dictionary)))
        print('Indexing Data...')
        for split in args['splits']:
            self.dataset[split].set_dictionary(self.dictionary)
            self.dataset[split].index_dataset()
        print('Indexing Done.')
        self.seed = args['seed'] if 'seed' in args else 0
        self.batch_size = args['batch_size']
        self.maxlen = args['maxlen']

    def get_dictionary(self):
        return self.dictionary

    def build_dictionary(self, raw_dataset, threshold=-1, nwords=-1):
        for sent in raw_dataset:
            for token in sent:
                self.dictionary.add_symbol(token)
        self.dictionary.finalize(threshold, nwords)

    def get_iterator(self, split, epoch=0, shuffle=True):
        dataset = self.dataset[split]
        idx = list(range(len(dataset)))
        if shuffle:
            np.random.seed(self.seed + epoch)
            np.random.shuffle(idx)
        return LanguageBatchIterator(idx,
                                     dataset,
                                     self.batch_size if split == 'train' else 1,
                                     self.maxlen)


class LaugnageDatasets(Dataset):

    def __init__(self, path, maxlen=0):
        super().__init__()
        self.path = path
        self.maxlen = maxlen
        self.raw_dataset = self._load_data()
        self.dictionary = None
        self.indexed_dataset = None

    def get_raw_dataset(self):
        return self.raw_dataset

    def _load_data(self):
        raw_dataset = []
        with open(self.path, 'r') as fin:
            for line in fin:
                sent = line.strip('\n').split()
                if self.maxlen == 0 or len(sent) <= self.maxlen:
                    raw_dataset.append(sent)
        return raw_dataset

    def index_dataset(self):
        self.indexed_dataset = []
        for sent in self.raw_dataset:
            self.indexed_dataset.append(Tokenizer.tokenize(sent, self.dictionary))

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, index):
        return {'id': index,
                'raw_sent': self.raw_dataset[index],
                'indexed_sent': self.indexed_dataset[index]}

    def set_dictionary(self, dictionary):
        self.dictionary = dictionary

    def get_dictionary(self):
        return self.dictionary


class LanguageBatchIterator():

    def __init__(self, iterable, dataset, batch_size, maxlen):
        self.iterable = iterable
        self.dataset = dataset
        self.dictionary = dataset.get_dictionary()
        self.padding_idx = self.dictionary.pad()
        self.eos_idx = self.dictionary.eos()
        self.count = 0
        self.itr = iter(self)
        self.batch_size = batch_size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.iterable) // self.batch_size + 1

    def __iter__(self):
        samples = []
        for idx in self.iterable:
            samples.append(self.dataset[idx])
            if len(samples) == self.batch_size:
                self.count += 1
                samples.sort(key=lambda x: len(x['indexed_sent']), reverse=True)
                yield samples, self._make_batch(samples)
                samples = []
        if len(samples) > 0:
            self.count += 1
            samples.sort(key=lambda x: len(x['indexed_sent']), reverse=True)
            yield samples, self._make_batch(samples)

    def _make_batch(self, samples):
        src = list(map(lambda x: x['indexed_sent'], samples))
        tgt = list(map(lambda x: x + [self.eos_idx], src))
        src_lengths = list(map(len, src))
        maxlen = max(src_lengths) if self.maxlen == 0 else self.maxlen
        src_batch = list(map(lambda x: x + [self.dictionary.pad()] * (maxlen - len(x)), src))
        tgt_lengths = list(map(len, tgt))
        maxlen = max(tgt_lengths) if self.maxlen == 0 else self.maxlen
        tgt_batch = list(map(lambda x: x + [self.dictionary.pad()] * (maxlen - len(x)), tgt))
        src = torch.LongTensor(src_batch)
        tgt = torch.LongTensor(tgt_batch)
        src_lengths = torch.LongTensor(src_lengths)
        tgt_lengths = torch.LongTensor(tgt_lengths)
        source_non_pad_mask = src.ne(self.padding_idx).float()
        target_non_pad_mask = tgt.ne(self.padding_idx).float()
        return {'src': src.cuda(),
                'src_lengths': src_lengths.cuda(),
                'src_non_pad_mask': source_non_pad_mask.cuda(),
                'tgt': tgt.cuda(),
                'tgt_lengths': tgt_lengths,
                'tgt_non_pad_mask': target_non_pad_mask.cuda(), }

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def get_progress(self):
        return self.count / len(self)
