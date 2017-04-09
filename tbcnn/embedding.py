from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_fold as td

from .config import param, hyper
from . import data


def linear_combine(clen, pclen, idx):
    Wl = param.get('Wl')
    Wr = param.get('Wr')

    dim = tf.unstack(tf.shape(Wl))[0]
    batch_shape = tf.shape(clen)

    f = (clen / pclen)
    l = (pclen - idx - 1) / (pclen - 1)
    r = (idx) / (pclen - 1)
    # when pclen == 1, replace nan items with 0.5
    l = tf.where(tf.is_nan(l), tf.ones_like(l) * 0.5, l)
    r = tf.where(tf.is_nan(r), tf.ones_like(r) * 0.5, r)

    lb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * l)
    rb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * r)
    fb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * f)

    lb = tf.reshape(lb, [-1, hyper.word_dim])
    rb = tf.reshape(rb, [-1, hyper.word_dim])

    tmp = tf.matmul(lb, Wl) + tf.matmul(rb, Wr)

    tmp = tf.reshape(tmp, [-1, hyper.word_dim, hyper.word_dim])

    return tf.matmul(fb, tmp)


def batch_mul(batch, weight):
    batch = tf.expand_dims(batch, axis=1)
    return tf.matmul(batch, weight)


def expand_dim_blk(axis):
    return td.Function(lambda tensor: tf.expand_dims(tensor, axis=axis))


def linear_combine_blk():
    blk = td.Function(linear_combine, infer_output_type=False)
    blk.set_output_type(td.TensorType([hyper.word_dim, hyper.word_dim]))
    return blk


def continous_weighted_add_blk():
    block = td.Composition(name='continous_weighted_add')
    with block.scope():
        initial = td.GetItem(0).reads(block.input)
        cur = td.GetItem(1).reads(block.input)

        last = td.GetItem(0).reads(initial)
        last = expand_dim_blk(axis=1).reads(last)
        idx = td.GetItem(1).reads(initial)

        cur_fea = td.GetItem(0).reads(cur)
        cur_clen = td.GetItem(1).reads(cur)
        pclen = td.GetItem(2).reads(cur)

        Wi = linear_combine_blk().reads(cur_clen, pclen, idx)

        weighted_fea = td.Function(batch_mul).reads(cur_fea, Wi)
        weighted_fea.set_output_type(td.TensorType([1, hyper.word_dim]))

        block.output.reads(
            td.Function(lambda a, b: tf.squeeze(tf.add(a, b), axis=1),
                        name='add_last_weighted_fea').reads(last, weighted_fea),
            # XXX: rewrite using tf.range
            td.Function(tf.add, name='add_idx_1').reads(idx,
                                                        td.FromTensor(tf.constant(1.)))
        )
    return block


def direct_embed_blk():
    return (td.InputTransform(lambda node: data.word2int[node['name']]) >> td.Scalar('int32')
            >> td.Function(param.get_embedding()))


def composed_embed_blk():
    leaf_case = direct_embed_blk()
    nonleaf_case = td.Composition(name='composed_embed_nonleaf')
    with nonleaf_case.scope():
        children = td.GetItem('children').reads(nonleaf_case.input)
        clen = td.Scalar().reads(td.GetItem('clen').reads(nonleaf_case.input))
        cclens = td.Map(td.GetItem('clen') >> td.Scalar()).reads(children)
        fchildren = td.Map(direct_embed_blk()).reads(children)

        initial_state = td.Composition()
        with initial_state.scope():
            initial_state.output.reads(
                td.FromTensor(tf.zeros(hyper.word_dim)),
                td.FromTensor(tf.zeros([])),
            )
        summed = td.Zip().reads(fchildren, cclens, td.Broadcast().reads(clen))
        summed = td.Fold(continous_weighted_add_blk(), initial_state).reads(summed)[0]
        added = td.Function(tf.add, name='add_bias').reads(summed, td.FromTensor(param.get('B')))
        relu = td.Function(tf.nn.relu).reads(added)
        nonleaf_case.output.reads(relu)

    return td.OneOf(lambda node: node['clen'] == 0,
                    {True: leaf_case, False: nonleaf_case})


def batch_nn_l2loss(a, b):
    """L2 loss between a and b, similar to tf.nn.l2_loss, but treat dim 0 as batch dim"""
    diff = tf.subtract(a, b)
    diff = tf.multiply(diff, diff)
    s = tf.reduce_sum(diff, axis=1)
    s = s / 2
    return s


def l2loss_blk():
    # rewrite using metric
    leaf_case = td.Composition()
    with leaf_case.scope():
        leaf_case.output.reads(td.FromTensor(tf.constant(1.)))
    nonleaf_case = td.Composition()
    with nonleaf_case.scope():
        direct = direct_embed_blk().reads(nonleaf_case.input)
        com = composed_embed_blk().reads(nonleaf_case.input)
        loss = td.Function(batch_nn_l2loss).reads(direct, com)
        nonleaf_case.output.reads(loss)
    return td.OneOf(lambda node: node['clen'] != 0,
                    {False: leaf_case, True: nonleaf_case})


def main():
    # traverse the tree to sum up the loss
    tree_sum_fwd = td.ForwardDeclaration(td.PyObjectType(), td.TensorType([]))
    tree_sum = td.Composition()
    with tree_sum.scope():
        myloss = l2loss_blk().reads(tree_sum.input)
        children = td.GetItem('children').reads(tree_sum.input)

        mapped = td.Map(tree_sum_fwd()).reads(children)
        summed = td.Reduce(td.Function(tf.add)).reads(mapped)
        summed = td.Function(tf.add).reads(summed, myloss)
        tree_sum.output.reads(summed)
    tree_sum_fwd.resolve_to(tree_sum)

    # Compile the block
    compiler = td.Compiler.create(tree_sum)
    loss = compiler.output_tensors
    train_step = tf.train.AdamOptimizer(learning_rate=0.2).minimize(loss)

    # load data node to record
    nodes = data.load()

    # train loop
    train_set = compiler.build_loom_inputs(nodes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf.summary.FileWriter('./tf_graph', graph=sess.graph)
        for epoch, shuffled in enumerate(td.epochs(train_set, hyper.num_epochs), 1):
            for batch in td.group_by_batches(shuffled, hyper.batch_size):
                train_feed_dict = {compiler.loom_input_tensor: batch}
                _, batch_loss = sess.run([train_step, loss], train_feed_dict)
                print(batch_loss)


if __name__ == '__main__':
    main()
