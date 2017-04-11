from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from ddt import ddt, data, unpack
import numpy.testing as nptest

import tensorflow as tf
import tensorflow_fold as td
import numpy as np

from tbcnn import embedding
from tbcnn.data import load as data_load


def linear_combine_np(clen, pclen, idx, Wl, Wr):
    if pclen == 1:
        l = r = 0.5
    else:
        l = (pclen - idx - 1) / (pclen - 1)
        r = idx / (pclen - 1)
    f = clen / pclen
    return f * (l * Wl + r * Wr)


@ddt
class TestEmbedding(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

        embedding.hyper.initialize(from_cmd=False, word_dim=3)
        embedding.param.initialize_embedding_weights()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    def _load_test_data(self):
        nodes, word2int = data_load('data/unittest.obj')
        return nodes[0], word2int

    def _get_leaf(self, root):
        for c in root['children']:
            return self._get_leaf(c)
        return root

    @unpack
    @data([1., 1., 0.], [1., 3., 0.], [2., 3., 1.], [0., 3., 2.])
    def test_linear_combine(self, clen, pclen, idx):
        """Test linear_combine_blk on data"""
        Wl = self.sess.run(embedding.param.get('Wl'))
        Wr = self.sess.run(embedding.param.get('Wr'))

        actual = (td.Scalar(), td.Scalar(), td.Scalar()) >> embedding.linear_combine_blk()
        actual = actual.eval((clen, pclen, idx), session=self.sess)

        desired = linear_combine_np(clen, pclen, idx, Wl, Wr)
        nptest.assert_allclose(actual, desired)

    def test_direct_embed_leaf(self):
        """Test direct_embed_blk on leaf node"""
        root, _ = self._load_test_data()
        leaf = self._get_leaf(root)

        actual = embedding.direct_embed_blk().eval(leaf, session=self.sess)

        # must goes after actual, otherwise weights are not created yet
        we = embedding.param.get('We').eval(session=self.sess)
        desired = we[leaf['name'], :]

        nptest.assert_allclose(actual, desired)

    def test_direct_embed_nonleaf(self):
        """Test direct_embed_blk on nonleaf node"""
        root, _ = self._load_test_data()

        actual = embedding.direct_embed_blk().eval(root, session=self.sess)

        # must goes after actual, otherwise weights are not created yet
        we = embedding.param.get('We').eval(session=self.sess)
        desired = we[root['name'], :]

        nptest.assert_allclose(actual, desired)

    def test_composed_embed(self):
        root, _ = self._load_test_data()

        Wl = embedding.param.get('Wl').eval(session=self.sess)
        Wr = embedding.param.get('Wr').eval(session=self.sess)
        B = embedding.param.get('B').eval(session=self.sess)

        actual = embedding.composed_embed_blk().eval(root, session=self.sess)

        pclen = root['clen']
        desired = np.zeros_like(actual)
        for idx, c in enumerate(root['children']):
            weight = linear_combine_np(c['clen'], pclen, idx, Wl, Wr)
            fc = embedding.direct_embed_blk().eval(c, session=self.sess)
            desired += np.matmul(fc, weight)
        desired += B

        if embedding.hyper.use_relu:
            desired[desired < 0] = 0
        else:
            desired = np.tanh(desired)

        nptest.assert_allclose(actual, desired, rtol=1e-6)

    def test_l2loss(self):
        root, _ = self._load_test_data()

        actual = embedding.l2loss_blk().eval(root, session=self.sess)

        direct = embedding.direct_embed_blk().eval(root, session=self.sess)
        com = embedding.composed_embed_blk().eval(root, session=self.sess)
        desired = tf.nn.l2_loss(direct - com).eval(session=self.sess)

        nptest.assert_allclose(actual, desired)

    def test_tree_sum(self):
        root, _ = self._load_test_data()
        actual = embedding.tree_sum_blk(embedding.l2loss_blk).eval(root, session=self.sess)

        loss = embedding.l2loss_blk()

        def traverse(root):
            running_total = loss.eval(root, session=self.sess)
            for c in root['children']:
                running_total += traverse(c)
            return running_total

        desired = traverse(root)

        nptest.assert_allclose(actual, desired)


if __name__ == '__main__':
    unittest.main()
