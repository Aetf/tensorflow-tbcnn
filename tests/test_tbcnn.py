from __future__ import absolute_import, division, print_function

import unittest
from ddt import ddt, data, unpack
import numpy.testing as nptest

import tensorflow as tf
import tensorflow_fold as td
import numpy as np

from tbcnn import embedding
from tbcnn.config import change_hyper
from tbcnn.data import word2int


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
        change_hyper(word_dim=3)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    @unpack
    @data([1., 1., 0.], [1., 3., 0.], [2., 3., 1.], [0., 3., 2.])
    def test_linear_combine(self, clen, pclen, idx):
        Wl = self.sess.run(embedding.param.get('Wl'))
        Wr = self.sess.run(embedding.param.get('Wr'))

        actual = (td.Scalar(), td.Scalar(), td.Scalar()) >> embedding.linear_combine_blk()
        actual = actual.eval((clen, pclen, idx), session=self.sess)

        desired = linear_combine_np(clen, pclen, idx, Wl, Wr)
        nptest.assert_allclose(actual, desired)

    def test_direct_embed(self):
        leaf = {'childcase': 0, 'children': [], 'clen': 0, 'name': 'String'}

        actual = embedding.direct_embed_blk().eval(leaf, session=self.sess)

        idx = word2int[leaf['name']]
        desired = embedding.param.get_embedding().weights.eval(session=self.sess)[idx, :]

        nptest.assert_allclose(actual, desired)

    def test_composed_embed(self):
        lv3 = {'childcase': 2,
               'children': [{'childcase': 1, 'children':
                             [{'childcase': 0, 'children': [], 'clen': 0, 'name': 'String'}],
                             'clen': 1, 'name': 'ColumnRef'},
                            {'childcase': 0, 'children': [],
                             'clen': 0, 'name': 'String'},
                            {'childcase': 1, 'children':
                             [{'childcase': 0, 'children': [], 'clen': 0, 'name': 'Integer'}],
                             'clen': 1, 'name': 'A_Const'}], 'clen': 3, 'name': 'A_Expr'}
        Wl = embedding.param.get('Wl').eval(session=self.sess)
        Wr = embedding.param.get('Wr').eval(session=self.sess)
        B = embedding.param.get('B').eval(session=self.sess)

        actual = embedding.composed_embed_blk().eval(lv3, session=self.sess)

        pclen = lv3['clen']
        desired = np.zeros_like(actual)
        for idx, c in enumerate(lv3['children']):
            weight = linear_combine_np(c['clen'], pclen, idx, Wl, Wr)
            fc = embedding.direct_embed_blk().eval(c, session=self.sess)
            desired += np.matmul(fc, weight)
        desired += B
        desired[desired < 0] = 0

        nptest.assert_allclose(actual, desired)

    def test_l2loss(self):
        lv3 = {'childcase': 2,
               'children': [{'childcase': 1, 'children':
                             [{'childcase': 0, 'children': [], 'clen': 0, 'name': 'String'}],
                             'clen': 1, 'name': 'ColumnRef'},
                            {'childcase': 0, 'children': [],
                             'clen': 0, 'name': 'String'},
                            {'childcase': 1, 'children':
                             [{'childcase': 0, 'children': [], 'clen': 0, 'name': 'Integer'}],
                             'clen': 1, 'name': 'A_Const'}], 'clen': 3, 'name': 'A_Expr'}

        actual = embedding.l2loss_blk().eval(lv3, session=self.sess)

        direct = embedding.direct_embed_blk().eval(lv3, session=self.sess)
        com = embedding.composed_embed_blk().eval(lv3, session=self.sess)
        desired = tf.nn.l2_loss(direct - com).eval(session=self.sess)

        nptest.assert_allclose(actual, desired)


if __name__ == '__main__':
    unittest.main()
