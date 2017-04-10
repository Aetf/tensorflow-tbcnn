from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from ddt import ddt, data, unpack
import numpy.testing as nptest

import tensorflow as tf
import tensorflow_fold as td
import numpy as np

from tbcnn import tbcnn
from tbcnn.data import load as data_load


@ddt
class TestTbcnn(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

        tbcnn.hyper.initialize(from_cmd=False, word_dim=3)
        tbcnn.param.initialize_tbcnn_weights()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    def _load_test_data(self):
        nodes, word2int = data_load('data/tests.obj')
        return nodes[0], word2int

    def _get_leaf(self, root):
        for c in root['children']:
            return self._get_leaf(c)
        return root

    @unpack
    @data([(1, ), 1.],
          [(2, 2), np.identity(2, 'float32')])
    def test_identity_initializer(self, shape, desired):
        initializer = tbcnn.identity_initializer()

        actual = initializer(shape).eval(session=self.sess)
        nptest.assert_allclose(actual, desired)

    def test_coding(self):
        root, _ = self._load_test_data()

        actual = tbcnn.coding_blk().eval(root, session=self.sess)

        direct = tbcnn.embedding.direct_embed_blk().eval(root, session=self.sess)
        com = tbcnn.embedding.composed_embed_blk().eval(root, session=self.sess)
        Wcomb1 = tbcnn.param.get('Wcomb1').eval(session=self.sess)
        Wcomb2 = tbcnn.param.get('Wcomb2').eval(session=self.sess)
        desired = np.matmul(direct, Wcomb1) + np.matmul(com, Wcomb2)

        nptest.assert_allclose(actual, desired)


if __name__ == '__main__':
    unittest.main()
