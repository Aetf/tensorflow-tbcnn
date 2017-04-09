from __future__ import absolute_import, division, print_function

import pickle

words = [
    'String', 'Integer', 'SelectStmt', 'A_Expr', 'ColumnRef', 'ResTarget', 'A_Const', 'RangeVar'
]
word2int = {w: idx for idx, w in enumerate(words)}


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = []
        self.type = ''


class NodeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'Node':
            return Node
        else:
            return super().find_class(module, name)


def node2dic(node):
    dic = {}
    dic['name'] = node.type
    dic['clen'] = len(node.children)
    dic['childcase'] = dic['clen']
    if dic['childcase'] >= 2:
        dic['childcase'] = 2
    dic['children'] = [node2dic(n) for n in node.children]
    return dic


def load(filename=None):
    if filename is None:
        filename = 'data/nodes.obj'
    with open(filename, 'rb') as f:
        nodes = NodeUnpickler(f).load()
    return [node2dic(n) for n in nodes]
