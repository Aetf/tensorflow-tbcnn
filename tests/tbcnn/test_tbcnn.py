from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from ddt import ddt, data, unpack
import numpy.testing as nptest

import tensorflow as tf
import tensorflow_fold as td
import numpy as np

from tbcnn import tbcnn
from tbcnn.data import load as data_load


def tri_combined_np(idx, pclen, depth, max_depth, Wconvl, Wconvr, Wconvt):
    t = (max_depth - depth) / max_depth
    if pclen == 1:
        r = (1 - t) * 0.5
    else:
        r = (1 - t) * (idx - 1) / (pclen - 1)
    l = (1 - t) * (1 - r)
    return l * Wconvl + r * Wconvr + t * Wconvt


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
        nodes, word2int = data_load('data/unittest.obj')
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

    @unpack
    @data([1., 1., 0., 2.],
          [1., 3., 1., 2.], [2., 3., 1., 2.], [3., 3., 1., 2.],
          [1., 1., 2., 2.], [1., 3., 2., 2.], [2., 3., 2., 2.], [3., 3., 2., 2.],
          [1., 2., 2., 2.], [2., 2., 2., 2.])
    def test_tri_combined(self, idx, pclen, depth, max_depth):
        """Test linear_combine_blk on data"""
        Wconvl = self.sess.run(tbcnn.param.get('Wconvl'))
        Wconvr = self.sess.run(tbcnn.param.get('Wconvr'))
        Wconvt = self.sess.run(tbcnn.param.get('Wconvt'))

        actual = (td.Scalar(), td.Scalar(), td.Scalar(), td.Scalar()) >> tbcnn.tri_combined_blk()
        actual = actual.eval((idx, pclen, depth, max_depth), session=self.sess)

        desired = tri_combined_np(idx, pclen, depth, max_depth, Wconvl, Wconvr, Wconvt)
        nptest.assert_allclose(actual, desired)

    def test_weighted_feature(self):
        root, _ = self._load_test_data()
        Wconvl = self.sess.run(tbcnn.param.get('Wconvl'))
        Wconvr = self.sess.run(tbcnn.param.get('Wconvr'))
        Wconvt = self.sess.run(tbcnn.param.get('Wconvt'))
        idx, pclen, depth, max_depth = (1., 1., 0., 2.)

        feature = tbcnn.coding_blk().eval(root, session=self.sess)

        actual = (td.Vector(feature.size), td.Scalar(),
                  td.Scalar(), td.Scalar(), td.Scalar()) >> tbcnn.weighted_feature_blk()
        actual = actual.eval((feature, idx, pclen, depth, max_depth), session=self.sess)

        desired = np.matmul(feature,
                            tri_combined_np(idx, pclen, depth, max_depth, Wconvl, Wconvr, Wconvt))

        nptest.assert_allclose(actual, desired)

    def test_feature_detector(self):
        root, _ = self._load_test_data()
        Wconvl = self.sess.run(tbcnn.param.get('Wconvl'))
        Wconvr = self.sess.run(tbcnn.param.get('Wconvr'))
        Wconvt = self.sess.run(tbcnn.param.get('Wconvt'))
        Bconv = self.sess.run(tbcnn.param.get('Bconv'))

        actual = tbcnn.feature_detector_blk().eval(root, session=self.sess)

        patch = tbcnn.collect_node_for_conv_patch_blk().eval(root, session=self.sess)
        desired = np.zeros_like(actual)
        for node, idx, pclen, depth, max_depth in patch:
            feature = tbcnn.coding_blk().eval(node, session=self.sess)
            desired += np.matmul(feature,
                                 tri_combined_np(idx, pclen, depth, max_depth, Wconvl, Wconvr, Wconvt))
        desired += Bconv
        desired = np.tanh(desired)

        print(actual)
        print(desired)

        nptest.assert_allclose(actual, desired, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
