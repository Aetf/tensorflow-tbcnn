from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_fold as td


# Hyper parameters
class hyper(object):
    # shapes
    word_dim = 10  # dimension of the feature vector for each node
    node_type_num = 20  # total number of node types
    # learning
    learning_rate = 0.2  # learning rate
    batch_size = 30
    num_epochs = 5


# weights management
class _paramobj(object):
    def __init__(self):
        self.weights = {}
        self.embedding_layer = None
        self.initialize()

    def create_variable(self, name, shape, initializer, dtype=tf.float32):
        self.weights[name] = tf.Variable(initializer(shape, dtype), name=name)

    def __getitem__(self, name):
        return self.weights[name]

    def initialize(self):
        self.create_variable('Wl', (hyper.word_dim, hyper.word_dim),
                             tf.random_uniform_initializer(-.2, .2))
        self.create_variable('Wr', (hyper.word_dim, hyper.word_dim),
                             tf.random_uniform_initializer(-.2, .2))
        self.create_variable('B', (hyper.word_dim,),
                             tf.random_uniform_initializer(-.2, .2))

    def get_embedding(self):
        if self.embedding_layer is None:
            self.embedding_layer = td.Embedding(hyper.node_type_num, hyper.word_dim)
        return self.embedding_layer


class param(object):
    _param = _paramobj()

    @classmethod
    def create_variable(clz, *args, **kwargs):
        return clz._param.create_variable(*args, **kwargs)

    @classmethod
    def get_embedding(clz):
        return clz._param.get_embedding()

    @classmethod
    def get(clz, name):
        return clz._param[name]


def change_hyper(**kwargs):
    """Change hyper parameters on the fly, and recreate param object. Must be the first function
    call. Only intended for testing code. Direct modify hyper in the source file for normal training.
    """
    for k, v in kwargs.items():
        setattr(hyper, k, v)
    param._param = _paramobj()
