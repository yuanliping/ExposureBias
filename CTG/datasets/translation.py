from CTG.tokenizer import Tokenizer
from CTG.dictionary import Dictionary

import numpy as np

import torch
from torch.utils.data import Dataset


class TranslationTask():

    def __init__(self, args):
        self.source, self.target = args['source'], args['target']
        self.dataset = {}

        print('Loading Data...')
        for split in args['splits']:
            if split == 'train':
                self.dataset[split] = TranslationDatasets('{}/{}'.format(args['data'], split),
                                                          self.source, self.target,
                                                          True,
                                                          maxlen=args['maxlen'])
            else:
                self.dataset[split] = TranslationDatasets('{}/{}'.format(args['data'], split),
                                                          self.source, self.target,
                                                          False,
                                                          maxlen=args['maxlen'])
        print('Loading Done.')

        print('Building Dictionary...')
        self.src_dict = Dictionary()
        if args['share_vocab']:
            self.tgt_dict = self.src_dict
        else:
            self.tgt_dict = Dictionary()
        self.build_dictionary(self.dataset['train'].get_raw_dataset(),
                              threshold=args['threshold'],
                              nwords=args['nwords'])
        print('Building Done.')
        print('Source Dictionary Size: {}'.format(len(self.get_source_dictionary())))
        print('Target Dictionary Size: {}'.format(len(self.get_target_dictionary())))

        print('Indexing Data...')
        for split in args['splits']:
            self.dataset[split].set_source_dictionary(self.src_dict)
            self.dataset[split].set_target_dictionary(self.tgt_dict)
            self.dataset[split].index_dataset()
        print('Indexing Done.')

        self.seed = args['seed'] if 'seed' in args else 0
        self.batch_size = args['batch_size']
        self.max_tokens = args['max_tokens']
        self.maxlen = args['maxlen']

    def get_source_dictionary(self):
        return self.src_dict

    def get_target_dictionary(self):
        return self.tgt_dict

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def build_dictionary(self, raw_dataset, threshold=-1, nwords=-1):
        for src_sent, tgt_sent in raw_dataset:
            for token in src_sent:
                self.src_dict.add_symbol(token)
            for token in tgt_sent:
                self.tgt_dict.add_symbol(token)
        self.src_dict.finalize(threshold, nwords)
        self.tgt_dict.finalize(threshold, nwords)
        assert self.src_dict.pad() == self.tgt_dict.pad()
        assert self.src_dict.eos() == self.tgt_dict.eos()

    def get_iterator(self, split, epoch=0, shuffle=False, single_sample=False):
        dataset = self.dataset[split]
        idx = list(range(len(dataset)))
        if shuffle:
            np.random.seed(self.seed + epoch)
            np.random.shuffle(idx)
        return LanguageBatchIterator(idx,
                                     dataset,
                                     1 if single_sample else self.batch_size,
                                     self.max_tokens,
                                     self.maxlen)


class TranslationDatasets(Dataset):

    def __init__(self, path, source, target, clip=True, maxlen=0):
        super().__init__()
        self.path = path
        self.clip = clip
        self.source, self.target = source, target
        self.maxlen = maxlen
        self.raw_dataset = None
        self._load_data()
        self.src_dict, self.tgt_dict = {}, {}
        self.indexed_dataset = {}

    def get_raw_dataset(self):
        return self.raw_dataset

    def _load_data(self):
        raw_dataset = {}
        for lang in [self.source, self.target]:
            raw_dataset[lang] = []
            with open('{}.{}'.format(self.path, lang), 'r') as fin:
                for line in fin:
                    sent = line.strip('\n').split()
                    raw_dataset[lang].append(sent)
        self.raw_dataset = []
        for src_sent, tgt_sent in zip(raw_dataset[self.source], raw_dataset[self.target]):
            if self.clip:
                if (self.maxlen == 0 or (len(src_sent) <= self.maxlen and len(tgt_sent) <= self.maxlen)) \
                        and (len(src_sent) > 0 and len(tgt_sent) > 0):
                    self.raw_dataset.append((src_sent, tgt_sent))
            else:
                self.raw_dataset.append((src_sent, tgt_sent))

        self.raw_dataset.sort(key=lambda x: len(x[1]))
        print(len(self.raw_dataset))

    def index_dataset(self):
        self.indexed_dataset = []
        for src_sent, tgt_sent in self.raw_dataset:
            src_idx = Tokenizer.tokenize(src_sent, self.src_dict)
            tgt_idx = Tokenizer.tokenize(tgt_sent, self.tgt_dict)
            self.indexed_dataset.append((src_idx, tgt_idx))

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, index):
        return {'id': index,
                'raw_source_sent': self.raw_dataset[index][0],
                'raw_target_sent': self.raw_dataset[index][1],
                'indexed_source_sent': self.indexed_dataset[index][0],
                'indexed_target_sent': self.indexed_dataset[index][1], }

    def set_source_dictionary(self, dictionary):
        self.src_dict = dictionary

    def set_target_dictionary(self, dictionary):
        self.tgt_dict = dictionary

    def get_source_dictionary(self):
        return self.src_dict

    def get_target_dictionary(self):
        return self.tgt_dict


class LanguageBatchIterator():

    def __init__(self, iterable, dataset, batch_size, max_tokens, maxlen):
        self.iterable = iterable
        self.dataset = dataset
        self.src_dict = dataset.get_source_dictionary()
        self.tgt_dict = dataset.get_target_dictionary()
        self.padding_idx = self.src_dict.pad()
        self.eos_idx = self.src_dict.eos()
        self.count = 0
        self.itr = iter(self)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.maxlen = maxlen

    def __len__(self):
        return len(self.iterable) // self.batch_size + 1

    def __iter__(self):
        samples = []
        for idx in self.iterable:
            d = self.dataset[idx]
            if len(samples) < self.batch_size and len(d['indexed_source_sent']) * len(samples) < self.max_tokens:
                samples.append(d)
                continue
            self.count += 1
            samples.sort(key=lambda x: len(x['indexed_source_sent']), reverse=True)
            yield samples, self._make_batch(samples)
            samples = []
        if len(samples) > 0:
            self.count += 1
            samples.sort(key=lambda x: len(x['indexed_source_sent']), reverse=True)
            yield samples, self._make_batch(samples)

    def _make_batch(self, samples):
        src = list(map(lambda x: x['indexed_source_sent'], samples))
        tgt = list(map(lambda x: x['indexed_target_sent'] + [self.eos_idx], samples))
        src_lengths = list(map(len, src))
        maxlen = max(src_lengths)
        src_batch = list(map(lambda x: x + [self.padding_idx] * (maxlen - len(x)), src))
        tgt_lengths = list(map(len, tgt))
        maxlen = max(tgt_lengths)
        tgt_batch = list(map(lambda x: x + [self.padding_idx] * (maxlen - len(x)), tgt))
        src_batch = torch.cuda.LongTensor(src_batch)
        src_lengths = torch.cuda.LongTensor(src_lengths)
        tgt_batch = torch.cuda.LongTensor(tgt_batch)
        tgt_lengths = torch.cuda.LongTensor(tgt_lengths)
        source_non_pad_mask = src_batch.ne(self.padding_idx).float()
        target_non_pad_mask = tgt_batch.ne(self.padding_idx).float()
        return {'src': src_batch,
                'src_lengths': src_lengths,
                'src_non_pad_mask': source_non_pad_mask,
                'tgt': tgt_batch,
                'tgt_lengths': tgt_lengths,
                'tgt_non_pad_mask': target_non_pad_mask, }

    def __next__(self):
        return next(self.itr)
