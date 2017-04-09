from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (object)

import os
import argparse

import tensorflow as tf
import tensorflow_fold as td


# Hyper parameters
class hyper(object):
    # shapes
    word_dim = 40  # dimension of the feature vector for each node
    node_type_num = 20  # total number of node types
    # learning
    learning_rate = 0.0002  # learning rate
    batch_size = 128
    num_epochs = 50
    # directories
    log_dir = '/tmp/workspace/tf_log'
    train_dir = '/tmp/workspace/tf_log'

    @classmethod
    def initialize(clz, from_cmd=True, **kwargs):
        if from_cmd:
            parser = argparse.ArgumentParser()
            parser.add_argument('--work_dir',
                                help='directory for saving files, defaults to /tmp/workspace/tf',
                                default='/tmp/workspace/tflogs')
            parser.add_argument('--log_dir',
                                help='directory for tensorboard logs, defaults to WORK_DIR/logs')
            parser.add_argument('--train_dir',
                                help='directory for model checkpoints, defaults to WORK_DIR/model')
            parser.add_argument('--num_epochs', help='total number of epochs', type=int, default=50)
            parser.add_argument('--batch_size', help='batch size', type=int, default=128)
            parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.0002)
            parser.add_argument('--word_dim', help='dimension of node feature', type=int, default=40)
            parser.add_argument('--node_type_num', help='total number of node types', type=int, default=20)
            args = parser.parse_args()
            if not os.path.exists(args.work_dir):
                os.makedirs(args.work_dir)
            if args.log_dir is None:
                args.log_dir = os.path.join(args.work_dir, 'logs')
            if args.train_dir is None:
                args.train_dir = os.path.join(args.work_dir, 'model')

            for k, v in vars(args).items():
                setattr(clz, k, v)

        for k, v in kwargs.items():
            setattr(clz, k, v)

        param._param = _paramobj()


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
        self.create_variable('We', (hyper.node_type_num, hyper.word_dim),
                             tf.random_uniform_initializer(-.2, .2))

    def get_embedding(self):
        if self.embedding_layer is None:
            self.embedding_layer = td.Embedding(hyper.node_type_num, hyper.word_dim)
        return self.embedding_layer


class param(object):
    _param = None

    @classmethod
    def create_variable(clz, *args, **kwargs):
        return clz._param.create_variable(*args, **kwargs)

    @classmethod
    def get_embedding(clz):
        return clz._param.get_embedding()

    @classmethod
    def get(clz, name):
        return clz._param[name]
