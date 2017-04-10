from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import open

import os
import operator
from datetime import datetime
from timeit import default_timer

import tensorflow as tf
import tensorflow_fold as td
from tensorflow.contrib.tensorboard.plugins import projector

from . import data
from .config import hyper, param


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
    mul = tf.matmul(batch, weight)
    return tf.squeeze(mul, axis=1)


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
        idx = td.GetItem(1).reads(initial)

        cur_fea = td.GetItem(0).reads(cur)
        cur_clen = td.GetItem(1).reads(cur)
        pclen = td.GetItem(2).reads(cur)

        Wi = linear_combine_blk().reads(cur_clen, pclen, idx)

        weighted_fea = td.Function(batch_mul).reads(cur_fea, Wi)

        block.output.reads(
            td.Function(tf.add, name='add_last_weighted_fea').reads(last, weighted_fea),
            # XXX: rewrite using tf.range
            td.Function(tf.add, name='add_idx_1').reads(idx, td.FromTensor(tf.constant(1.)))
        )
    return block


def clip_by_norm_blk(norm=1.0):
    return td.Function(lambda x: tf.clip_by_norm(x, norm, axes=[1]))


def direct_embed_blk():
    return (td.GetItem('name') >> td.Scalar('int32')
            >> td.Function(lambda x: tf.nn.embedding_lookup(param.get('We'), x))
            >> clip_by_norm_blk())


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
        normed = clip_by_norm_blk().reads(added)
        relu = td.Function(tf.nn.relu).reads(normed)
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


# generalize to tree_reduce, accepts one block that takes two node, returns a value
def tree_sum_blk(loss_blk):
    # traverse the tree to sum up the loss
    tree_sum_fwd = td.ForwardDeclaration(td.PyObjectType(), td.TensorType([]))
    tree_sum = td.Composition()
    with tree_sum.scope():
        myloss = loss_blk().reads(tree_sum.input)
        children = td.GetItem('children').reads(tree_sum.input)

        mapped = td.Map(tree_sum_fwd()).reads(children)
        summed = td.Reduce(td.Function(tf.add)).reads(mapped)
        summed = td.Function(tf.add).reads(summed, myloss)
        tree_sum.output.reads(summed)
    tree_sum_fwd.resolve_to(tree_sum)
    return tree_sum


def write_embedding_metadata(writer, word2int):
    metadata_path = os.path.join(hyper.train_dir, 'embedding_meta.tsv')
    # dump embedding mapping
    items = sorted(word2int.items(), key=operator.itemgetter(1))
    with open(metadata_path, 'w') as f:
        for item in items:
            print(item[0], file=f)

    config = projector.ProjectorConfig()
    config.model_checkpoint_dir = hyper.train_dir  # not work yet. TF doesn't support model_checkpoint_dir
    embedding = config.embeddings.add()
    embedding.tensor_name = param.get('We').name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata_path
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)


def main():
    hyper.initialize(variable_scope='embedding')

    # create model variables
    param.initialize_embedding_weights()

    # Compile the block
    tree_sum = tree_sum_blk(l2loss_blk)
    compiler = td.Compiler.create(tree_sum)
    (batched_loss, ) = compiler.output_tensors
    loss = tf.reduce_mean(batched_loss)
    opt = tf.train.AdamOptimizer(learning_rate=hyper.learning_rate)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_step = opt.minimize(loss, global_step=global_step)

    # Attach summaries
    tf.summary.histogram('Wl', param.get('Wl'))
    tf.summary.histogram('Wr', param.get('Wr'))
    tf.summary.histogram('B', param.get('B'))
    tf.summary.histogram('Embedding', param.get('We'))
    tf.summary.scalar('loss', loss)

    summary_op = tf.summary.merge_all()

    # load data node to record
    nodes, word2int = data.load('data/nodes.obj')
    nodes_valid, word2int = data.load('data/valid_nodes.obj', word2int)
    nodes = nodes + nodes_valid

    # create missing dir
    if not os.path.exists(hyper.train_dir):
        os.makedirs(hyper.train_dir)

    # train loop
    saver = tf.train.Saver()
    train_set = compiler.build_loom_inputs(nodes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(hyper.log_dir, graph=sess.graph)
        write_embedding_metadata(summary_writer, word2int)

        for epoch, shuffled in enumerate(td.epochs(train_set, hyper.num_epochs), 1):
            for step, batch in enumerate(td.group_by_batches(shuffled, hyper.batch_size), 1):
                train_feed_dict = {compiler.loom_input_tensor: batch}

                start_time = default_timer()
                _, loss_value, summary, gstep = sess.run([train_step, loss, summary_op, global_step], train_feed_dict)
                duration = default_timer() - start_time

                print('{}: global {} epoch {}, step {}, loss = {:.2f} ({:.1f} samples/sec; {:.3f} sec/batch)'
                      .format(datetime.now(), gstep, epoch, step, loss_value,
                              hyper.batch_size / duration, duration))
                if gstep % 10 == 0:
                    summary_writer.add_summary(summary, gstep)
                if gstep % 100 == 0:
                    saver.save(sess, os.path.join(hyper.train_dir, "model.ckpt"), global_step=gstep)


if __name__ == '__main__':
    main()
