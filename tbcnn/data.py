from __future__ import absolute_import, division, print_function

import pickle


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


def load(filename=None):
    if filename is None:
        filename = 'data/nodes.obj'
    with open(filename, 'rb') as f:
        nodes = NodeUnpickler(f).load()

    word2int = {}
    nodes = [node2dic(n, word2int) for n in nodes]
    return nodes, word2int
