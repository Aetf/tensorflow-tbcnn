from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import open, super, object
import six

import random
import pickle
import itertools
import logging

logger = logging.getLogger(__name__)


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.type = ''


if six.PY2:
    class NodeUnpickler(pickle.Unpickler, object):
        def find_class(self, module, name):
            if module == '__main__' and name == 'Node':
                return Node
            else:
                return super().find_class(module, name)
else:
    class NodeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__main__' and name == 'Node':
                return Node
            else:
                return super().find_class(module, name)


class DataSet(object):
    def __init__(self, word2int):
        self.word2int = word2int
        self.splits = {}

    def _add_split(self, name, length, gen):
        self.splits[name] = (length, gen)

    def get_split(self, split):
        """split = 'train' or 'val' or 'test'
        returns generator
        """
        if split == 'all':
            total_len = sum([l for l, _ in self.splits.values()])
            return total_len, itertools.chain(*[gen for _, gen in self.splits.values()])
        return self.splits[split]


def node2dic(node, word2int):
    dic = {}

    # convert string type to id
    if node.type in word2int:
        wid = word2int[node.type]
    else:
        wid = len(word2int)
        word2int[node.type] = wid
    dic['name'] = wid

    dic['clen'] = len(node.children)
    dic['childcase'] = dic['clen']
    if dic['childcase'] >= 2:
        dic['childcase'] = 2
    dic['children'] = [node2dic(n, word2int) for n in node.children]
    return dic


def load(filename=None, word2int=None):
    if filename is None:
        filename = 'data/raw/nodes.obj'
    with open(filename, 'rb') as f:
        nodes = NodeUnpickler(f).load()
    random.shuffle(nodes)

    if word2int is None:
        word2int = {}
    nodes = [node2dic(n, word2int) for n in nodes]
    return nodes, word2int


def _get_filename(basename, split):
    return '{}.{}.obj'.format(basename, split)


def convert_to_stream_dataset(basename, train_frac=0.6, val_frac=0.6):
    nodes, word2int = load('data/raw/nodes.obj')
    nodes_valid, word2int = load('data/raw/valid_nodes.obj', word2int)
    random.shuffle(nodes)
    random.shuffle(nodes_valid)

    # add label
    # label 1 means injection statement
    # label 0 means benign statement
    nodes = [(n, 1) for n in nodes]
    nodes_valid = [(n, 0) for n in nodes_valid]

    def split(items, frac1, frac2):
        p1 = int(len(items) * frac1)
        p2 = int(len(items) * (frac1 + frac2))
        return items[:p1], items[p1:p2], items[p2:]

    samples_train, samples_val, samples_test = split(nodes, train_frac, val_frac)
    split1, split2, split3 = split(nodes_valid, train_frac, val_frac)
    samples_train += split1
    samples_val += split2
    samples_test += split3

    random.shuffle(samples_train)
    random.shuffle(samples_val)
    random.shuffle(samples_test)

    protocol = 2  # Use protocol version 2 for compatible with python 2

    def write_to_file(samples, split):
        filepath = _get_filename(basename, split)
        with open(filepath, 'wb') as f:
            pickle.dump(len(samples), f, protocol=protocol)
            for sample in samples:
                pickle.dump(sample, f, protocol=protocol)

    with open(_get_filename(basename, 'meta'), 'wb') as f:
        pickle.dump(word2int, f, protocol=protocol)
        pickle.dump(['train', 'val', 'test'], f, protocol=protocol)
    write_to_file(samples_train, 'train')
    write_to_file(samples_val, 'val')
    write_to_file(samples_test, 'test')


def load_dataset(basename):
    with open(_get_filename(basename, 'meta'), 'rb') as f:
        word2int = pickle.load(f)
        splits = pickle.load(f)

    def loader(f):
        try:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
        finally:
            f.close()

    ds = DataSet(word2int)

    for split in splits:
        f = open(_get_filename(basename, split), 'rb')
        try:
            length = pickle.load(f)
        except:
            f.close()
            raise
        gen = loader(f)
        ds._add_split(split, length, gen)

    return ds
