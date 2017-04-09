# flake8: noqa
from __future__ import absolute_import, division, print_function

from .embedding import *


def find_leaf(root):
    for c in root['children']:
        return find_leaf(c)
    return root


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

direct = direct_embed()
com = composed_embed()

Wl = sess.run(param.get('Wl'))
Wr = sess.run(param.get('Wr'))
B = sess.run(param.get('B'))

root = data.load()[0]
c1 = root['children'][0]
c2 = root['children'][1]


leaf = find_leaf(root)
lv2 = root['children'][0]['children'][0]['children'][0]
