from CTG.tokenizer import Tokenizer
from CTG.dictionary import Dictionary

import numpy as np
import re

import torch
from torch.utils.data import Dataset


class ConsitituencyParsingTask(object):

    def __init__(self, args):
        self.source, self.target = args['source'], args['target']
        self.dataset = {}

        print('Loading Data...')
        for split in args['splits']:
            self.dataset[split] = ConsitituencyParsingDatasets('{}/{}'.format(args['data'], split),
                                                               self.source, self.target,
                                                               maxlen=args['maxlen'])
        print('Loading Done.')

        print('Building Dictionary...')
        self.src_dict = Dictionary()
        if args['share_vocab']:
            self.tgt_dict = self.src_dict
        else:
            self.tgt_dict = Dictionary()
        self.build_dictionary(self.dataset['train'].get_raw_dataset(), nwords=args['nwords'])
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
        self.maxlen = args['maxlen']

    def get_source_dictionary(self):
        return self.src_dict

    def get_target_dictionary(self):
        return self.tgt_dict

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def _merge(self, traverse, sent):
        tokens = traverse.split()
        words = sent.split()
        idx = 0
        for i, t in enumerate(tokens):
            if not t.startswith('(') and not t.endswith(')'):
                tokens[i] = t + ' ' + words[idx]
                idx += 1
            if t.endswith(')'):
                tokens[i] = re.sub('\).*', ')', t)
        traverse = ' '.join(tokens)
        return traverse

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

    def get_iterator(self, split, epoch=0, shuffle=False):
        dataset = self.dataset[split]
        idx = list(range(len(dataset)))
        if shuffle:
            np.random.seed(self.seed + epoch)
            np.random.shuffle(idx)
        return LanguageBatchIterator(idx,
                                     dataset,
                                     self.batch_size,
                                     self.maxlen)


class ConsitituencyParsingDatasets(Dataset):

    def __init__(self, path, source, target, maxlen=0):
        super().__init__()
        self.path = path
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
            if self.maxlen == 0 or (len(src_sent) < self.maxlen and len(tgt_sent) < self.maxlen):
                self.raw_dataset.append((src_sent, tgt_sent))
        self.raw_dataset.sort(key=lambda x: len(x[1]))

    def index_dataset(self):
        self.indexed_dataset = []
        for src_sent, tgt_sent in self.raw_dataset:
            self.indexed_dataset.append((Tokenizer.tokenize(src_sent, self.src_dict),
                                         Tokenizer.tokenize(tgt_sent, self.tgt_dict)))

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, index):
        return {'id': index,
                'raw_source_sent': self.raw_dataset[index][0],
                'raw_target_sent': self.raw_dataset[index][1],
                'indexed_source_sent': self.indexed_dataset[index][0],
                'indexed_target_sent': self.indexed_dataset[index][1],}

    def set_source_dictionary(self, dictionary):
        self.src_dict = dictionary

    def set_target_dictionary(self, dictionary):
        self.tgt_dict = dictionary

    def get_source_dictionary(self):
        return self.src_dict

    def get_target_dictionary(self):
        return self.tgt_dict


class LanguageBatchIterator():

    def __init__(self, iterable, dataset, batch_size, maxlen):
        self.iterable = iterable
        self.dataset = dataset
        self.src_dict = dataset.get_source_dictionary()
        self.tgt_dict = dataset.get_target_dictionary()
        self.padding_idx = self.src_dict.pad()
        self.eos_idx = self.src_dict.eos()
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
        src_batch = torch.LongTensor(src_batch)
        src_lengths = torch.LongTensor(src_lengths)
        tgt_batch = torch.LongTensor(tgt_batch)
        tgt_lengths = torch.LongTensor(tgt_lengths)
        source_non_pad_mask = src_batch.ne(self.padding_idx).float()
        target_non_pad_mask = tgt_batch.ne(self.padding_idx).float()
        return {'src': src_batch.cuda(),
                'src_lengths': src_lengths.cuda(),
                'src_non_pad_mask': source_non_pad_mask.cuda(),
                'tgt': tgt_batch.cuda(),
                'tgt_lengths': tgt_lengths.cuda(),
                'tgt_non_pad_mask': target_non_pad_mask.cuda(),}

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def get_progress(self):
        return self.count / len(self)
