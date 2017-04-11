from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (object)

import os
import argparse

import tensorflow as tf


# Hyper parameters
class hyper(object):
    # shapes
    node_type_num = 40  # total number of node types
    word_dim = 40  # dimension of the feature vector for each node
    conv_dim = 40  # number of conv feature detectors
    fc_dim = 1024  # dimension of fc output
    output_dim = 2  # number of output classes
    # structure
    use_relu = False
    # learning
    learning_rate = 0.0002  # learning rate
    batch_size = 128
    num_epochs = 50
    # directories
    log_dir = '/tmp/workspace/tf_log'
    train_dir = '/tmp/workspace/tf_log'
    embedding_dir = '/tmp/workspace/embedding_checkpoints/model'
    # variables
    variable_scope = ''

    @classmethod
    def initialize(clz, from_cmd=True, **kwargs):
        if from_cmd:
            parser = argparse.ArgumentParser()
            parser.add_argument('--embedding_dir',
                                help="""Saved embedding metrix,
                                defaults to /tmp/workspace/embedding_checkpoints/model
                                Only used when train tbcnn.
                                """,
                                default='/tmp/workspace/embedding_checkpoints/model')
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
            parser.add_argument('--word_dim', help='dimension of node feature', type=int, default=100)
            parser.add_argument('--conv_dim',
                                help='dimension of conv feature detectors', type=int, default=50)
            parser.add_argument('--fc_dim',
                                help='dimension of fc layer', type=int, default=1024)
            parser.add_argument('--output_dim',
                                help='number of output classes, default to 2. Should not need to change',
                                type=int, default=2)
            parser.add_argument('--node_type_num', help='total number of node types', type=int, default=20)
            parser.add_argument('--use_relu', type=bool, default=False)
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


# weights management
class param(object):
    @classmethod
    def create_variable(clz, name, shape, initializer, dtype=tf.float32, trainable=True):
        with tf.variable_scope(hyper.variable_scope):
            return tf.get_variable(name, shape=shape, dtype=dtype,
                                   trainable=trainable, initializer=initializer)

    @classmethod
    def get(clz, name):
        with tf.variable_scope(hyper.variable_scope, reuse=True):
            return tf.get_variable(name)

    @classmethod
    def initialize_embedding_weights(clz):
        clz.create_variable('Wl', (hyper.word_dim, hyper.word_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('Wr', (hyper.word_dim, hyper.word_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('B', (hyper.word_dim,),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('We', (hyper.node_type_num, hyper.word_dim),
                            tf.random_uniform_initializer(-.2, .2))

    @classmethod
    def initialize_tbcnn_weights(clz):
        clz.initialize_embedding_weights()
        # Don't train We
        tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES).remove(clz.get('We'))

        clz.create_variable('Wcomb1', (hyper.word_dim, hyper.word_dim),
                            tf.constant_initializer(-.2, .2))
        clz.create_variable('Wcomb2', (hyper.word_dim, hyper.word_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('Wconvt', (hyper.word_dim, hyper.conv_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('Wconvl', (hyper.word_dim, hyper.conv_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('Wconvr', (hyper.word_dim, hyper.conv_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('Bconv', (hyper.conv_dim,),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('FC1/weight', (hyper.conv_dim, hyper.fc_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('FC1/bias', (hyper.fc_dim,),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('FC2/weight', (hyper.fc_dim, hyper.output_dim),
                            tf.random_uniform_initializer(-.2, .2))
        clz.create_variable('FC2/bias', (hyper.output_dim, ),
                            tf.random_uniform_initializer(-.2, .2))
