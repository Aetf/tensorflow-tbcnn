from __future__ import absolute_import, division, print_function

import os
import logging
from timeit import default_timer

import numpy as np
import tensorflow as tf
import tensorflow_fold as td

from . import apputil
from . import data
from . import embedding
from .config import hyper, param

logger = logging.getLogger(__name__)


def identity_initializer():
    def _initializer(shape, dtype=np.float32):
        if len(shape) == 1:
            return tf.constant(1., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(array, dtype=dtype)
        else:
            raise
    return _initializer


def coding_blk():
    """Input: node dict
    Output: TensorType([1, hyper.word_dim])
    """
    Wcomb1 = param.get('Wcomb1')
    Wcomb2 = param.get('Wcomb2')

    blk = td.Composition()
    with blk.scope():
        direct = embedding.direct_embed_blk().reads(blk.input)
        composed = embedding.composed_embed_blk().reads(blk.input)
        Wcomb1 = td.FromTensor(param.get('Wcomb1'))
        Wcomb2 = td.FromTensor(param.get('Wcomb2'))

        direct = td.Function(embedding.batch_mul).reads(direct, Wcomb1)
        composed = td.Function(embedding.batch_mul).reads(composed, Wcomb2)

        added = td.Function(tf.add).reads(direct, composed)
        blk.output.reads(added)
    return blk


def collect_node_for_conv_patch_blk(max_depth=2):
    """Input: node dict
    Output: flattened list of all collected nodes, in the format
    [(node, idx, pclen, depth, max_depth), ...]
    """
    def _collect_patch(node):
        collected = [(node, 1, 1, 0, max_depth)]

        def recurse_helper(node, depth):
            if depth > max_depth:
                return
            for idx, c in enumerate(node['children'], 1):
                collected.append((c, idx, node['clen'], depth + 1, max_depth))
                recurse_helper(c, depth + 1)

        recurse_helper(node, 0)
        return collected

    return td.InputTransform(_collect_patch)


def tri_combined(idx, pclen, depth, max_depth):
    """TF function, input: idx, pclen, depth, max_depth as batch (1D Tensor)
    Output: weight tensor (3D Tensor), first dim is batch
    """
    Wconvt = param.get('Wconvt')
    Wconvl = param.get('Wconvl')
    Wconvr = param.get('Wconvr')

    dim = tf.unstack(tf.shape(Wconvt))[0]
    batch_shape = tf.shape(idx)

    tmp = (idx - 1) / (pclen - 1)
    # when pclen == 1, replace nan items with 0.5
    tmp = tf.where(tf.is_nan(tmp), tf.ones_like(tmp) * 0.5, tmp)

    t = (max_depth - depth) / max_depth
    r = (1 - t) * tmp
    l = (1 - t) * (1 - r)

    lb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * l)
    rb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * r)
    tb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * t)

    lb = tf.reshape(lb, [-1, dim])
    rb = tf.reshape(rb, [-1, dim])
    tb = tf.reshape(tb, [-1, dim])

    tmp = tf.matmul(lb, Wconvl) + tf.matmul(rb, Wconvr) + tf.matmul(tb, Wconvt)

    tmp = tf.reshape(tmp, [-1, hyper.word_dim, hyper.conv_dim])
    return tmp


def tri_combined_blk():
    blk = td.Function(tri_combined, infer_output_type=False)
    blk.set_output_type(td.TensorType([hyper.word_dim, hyper.conv_dim]))
    return blk


def weighted_feature_blk():
    """Input: (feature                       , idx   , pclen,  depth,  max_depth)
              (TensorType([hyper.word_dim, ]), Scalar, Scalar, Scalar, Scalar)
    Output: weighted_feature
            TensorType([hyper.conv_dim, ])
    """
    blk = td.Composition()
    with blk.scope():
        fea = blk.input[0]
        Wi = tri_combined_blk().reads(blk.input[1], blk.input[2], blk.input[3], blk.input[4])

        weighted_fea = td.Function(embedding.batch_mul).reads(fea, Wi)

        blk.output.reads(weighted_fea)
    return blk


def feature_detector_blk(max_depth=2):
    """Input: node dict
    Output: TensorType([hyper.conv_dim, ])
    Single patch of the conv. Depth is max_depth
    """
    blk = td.Composition()
    with blk.scope():
        nodes_in_patch = collect_node_for_conv_patch_blk(max_depth=max_depth).reads(blk.input)

        # map from python object to tensors
        mapped = td.Map(td.Record((coding_blk(), td.Scalar(), td.Scalar(),
                                   td.Scalar(), td.Scalar()))).reads(nodes_in_patch)
        # mapped = [(feature, idx, depth, max_depth), (...)]

        # compute weighted feature for each elem
        weighted = td.Map(weighted_feature_blk()).reads(mapped)
        # weighted = [fea, fea, fea, ...]

        # add together
        added = td.Reduce(td.Function(tf.add)).reads(weighted)
        # added = TensorType([hyper.conv_dim, ])

        # add bias
        biased = td.Function(tf.add).reads(added, td.FromTensor(param.get('Bconv')))
        # biased = TensorType([hyper.conv_dim, ])

        # tanh
        tanh = td.Function(tf.nn.tanh).reads(biased)
        # tanh = TensorType([hyper.conv_dim, ])

        blk.output.reads(tanh)
    return blk


# generalize to tree_fold, accepts one block that takes two node, returns a value
def dynamic_pooling_blk():
    """Input: root node dic
    Output: pooled, TensorType([hyper.conv_dim, ])
    """
    leaf_case = feature_detector_blk()

    pool_fwd = td.ForwardDeclaration(td.PyObjectType(), td.TensorType([hyper.conv_dim, ]))
    pool = td.Composition()
    with pool.scope():
        cur_fea = feature_detector_blk().reads(pool.input)
        children = td.GetItem('children').reads(pool.input)

        mapped = td.Map(pool_fwd()).reads(children)
        summed = td.Reduce(td.Function(tf.maximum)).reads(mapped)
        summed = td.Function(tf.maximum).reads(summed, cur_fea)
        pool.output.reads(summed)
    pool = td.OneOf(lambda x: x['clen'] == 0,
                    {True: leaf_case, False: pool})
    pool_fwd.resolve_to(pool)
    return pool


def build_model():
    # create model variables
    param.initialize_tbcnn_weights()

    # Compile the block and append fc layers
    tree_pooling = dynamic_pooling_blk()
    compiler = td.Compiler.create((tree_pooling, td.Scalar(dtype='int64')))
    (pooled, batched_labels) = compiler.output_tensors

    fc1 = tf.nn.relu(tf.add(tf.matmul(pooled, param.get('FC1/weight')), param.get('FC1/bias')))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, param.get('FC2/weight')), param.get('FC2/bias')))

    # our prediction output with accuracy calc
    logits = tf.nn.softmax(fc2)
    correct_prediction = tf.equal(tf.argmax(logits, 1), batched_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_size_op = tf.unstack(tf.shape(batched_labels))[0]

    return compiler, fc2, logits, batched_labels, accuracy, batch_size_op


def train_with_val(unscaled_logits, batched_labels, train_accuracy):
    # calculate weight decay loss
    decay_names = ['Wl', 'Wr', 'Wconvl', 'Wconvr', 'Wconvt']
    decay_loss = tf.reduce_sum(
        input_tensor=hyper.weight_decay * tf.stack([tf.nn.l2_loss(param.get(n)) for n in decay_names]),
        name='weights_norm')

    # Calculate loss and apply optimizer
    batched_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits,
                                                                  labels=batched_labels)
    loss = tf.reduce_mean(batched_loss) + decay_loss
    opt = tf.train.AdamOptimizer(learning_rate=hyper.learning_rate)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_step = opt.minimize(loss, global_step=global_step)

    # Attach summaries
    tf.summary.histogram('Wl', param.get('Wl'))
    tf.summary.histogram('Wr', param.get('Wr'))
    tf.summary.histogram('B', param.get('B'))
    tf.summary.histogram('Wconvl', param.get('Wconvl'))
    tf.summary.histogram('Wconvr', param.get('Wconvr'))
    tf.summary.histogram('Wconvt', param.get('Wconvt'))
    tf.summary.histogram('Bconv', param.get('Bconv'))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('train_accuracy', train_accuracy)
    summary_op = tf.summary.merge_all()

    return loss, global_step, train_step, summary_op


def do_train():
    # load data early to get node_type_num
    ds = data.load_dataset('data/statements')
    hyper.node_type_num = len(ds.word2int)

    (compiler, unscaled_logits, logits, batched_labels,
     raw_accuracy, batch_size_op) = build_model()

    (loss, global_step, train_step, summary_op) = train_with_val(unscaled_logits, batched_labels,
                                                                 raw_accuracy)

    val_summary_op = tf.summary.scalar('val_accuracy', raw_accuracy)

    # create missing dir
    if not os.path.exists(hyper.train_dir):
        os.makedirs(hyper.train_dir)

    # restorer for embedding matrix
    restorer = tf.train.Saver({'embedding/We': param.get('We')})
    embedding_path = tf.train.latest_checkpoint(hyper.embedding_dir)
    if embedding_path is None:
        raise ValueError('Path to embedding checkpoint is incorrect: ' + hyper.embedding_dir)

    # train loop
    saver = tf.train.Saver()
    train_set = compiler.build_loom_inputs(ds.get_split('train')[1])
    val_set = compiler.build_loom_inputs(ds.get_split('val')[1])
    with tf.Session() as sess:
        # Restore embedding matrix first
        restorer.restore(sess, embedding_path)
        # Initialize other variables
        gvariables = tf.global_variables()
        gvariables.remove(param.get('We'))  # exclude We
        sess.run(tf.variables_initializer(gvariables))

        summary_writer = tf.summary.FileWriter(hyper.log_dir, graph=sess.graph)

        val_step_counter = 0
        shuffled = zip(td.epochs(train_set, hyper.num_epochs),
                       td.epochs(val_set, hyper.num_epochs))
        for epoch, (train_shuffled, val_shuffled) in enumerate(shuffled, 1):
            for step, batch in enumerate(td.group_by_batches(train_shuffled, hyper.batch_size), 1):
                train_feed_dict = {compiler.loom_input_tensor: batch}

                start_time = default_timer()
                (_, loss_value, summary,
                 gstep, actual_bsize) = sess.run([train_step, loss, summary_op, global_step,
                                                  batch_size_op],
                                                 train_feed_dict)
                duration = default_timer() - start_time

                logger.info('global %d epoch %d step %d loss = %.2f (%.1f samples/sec; %.3f sec/batch)',
                            gstep, epoch, step, loss_value, actual_bsize / duration, duration)
                if gstep % 10 == 0:
                    summary_writer.add_summary(summary, gstep)

            # do a validation test
            logger.info('')
            logger.info('======================= Validation ====================================')
            accumulated_accuracy = 0.
            total_size = 0
            start_time = default_timer()
            for batch in td.group_by_batches(val_shuffled, hyper.batch_size):
                feed_dict = {compiler.loom_input_tensor: batch}
                accuracy_value, actual_bsize, val_summary = sess.run([raw_accuracy, batch_size_op,
                                                                      val_summary_op], feed_dict)
                summary_writer.add_summary(val_summary, val_step_counter)
                accumulated_accuracy += accuracy_value * actual_bsize
                total_size += actual_bsize
                val_step_counter += 1
                logger.info('validation step, accuracy = %.2f, current batch = %d, processed = %d',
                            accuracy_value, actual_bsize, total_size)
            duration = default_timer() - start_time
            total_accuracy = accumulated_accuracy / total_size
            logger.info('validation acc = %.2f%% (%.1f samples/sec; %.2f seconds)',
                        total_accuracy * 100, actual_bsize / duration, duration)
            saved_path = saver.save(sess, os.path.join(hyper.train_dir, "model.ckpt"), global_step=gstep)
            logger.info('validation saved path: %s', saved_path)
            logger.info('======================= Validation End =================================')
            logger.info('')


def do_evaluation():
    # load data early to get node_type_num
    ds = data.load_dataset('data/statements')
    hyper.node_type_num = len(ds.word2int)

    (compiler, _, _, _, raw_accuracy, batch_size_op) = build_model()

    # restorer for embedding matrix
    embedding_path = tf.train.latest_checkpoint(hyper.embedding_dir)
    if embedding_path is None:
        raise ValueError('Path to embedding checkpoint is incorrect: ' + hyper.embedding_dir)

    # restorer for other variables
    checkpoint_path = tf.train.latest_checkpoint(hyper.train_dir)
    if checkpoint_path is None:
        raise ValueError('Path to tbcnn checkpoint is incorrect: ' + hyper.train_dir)

    restored_vars = tf.get_collection_ref('restored')
    restored_vars.append(param.get('We'))
    restored_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    embeddingRestorer = tf.train.Saver({'embedding/We': param.get('We')})
    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # train loop
    total_size, test_gen = ds.get_split('test')
    test_set = compiler.build_loom_inputs(test_gen)
    with tf.Session() as sess:
        # Restore embedding matrix first
        embeddingRestorer.restore(sess, embedding_path)
        # Restore others
        restorer.restore(sess, checkpoint_path)
        # Initialize other variables
        gvariables = [v for v in tf.global_variables() if v not in tf.get_collection('restored')]
        sess.run(tf.variables_initializer(gvariables))

        for shuffled in td.epochs(test_set, 1):
            logger.info('')
            logger.info('======================= Evaluation ====================================')
            accumulated_accuracy = 0.
            start_time = default_timer()
            for step, batch in enumerate(td.group_by_batches(shuffled, hyper.batch_size), 1):
                feed_dict = {compiler.loom_input_tensor: batch}
                accuracy_value, actual_bsize = sess.run([raw_accuracy, batch_size_op], feed_dict)
                accumulated_accuracy += accuracy_value * actual_bsize
                logger.info('evaluation in progress: running accuracy = %.2f, processed = %d / %d',
                            accuracy_value, (step - 1) * hyper.batch_size + actual_bsize, total_size)
            duration = default_timer() - start_time
            total_accuracy = accumulated_accuracy / total_size
            logger.info('evaluation accumulated accuracy = %.2f%% (%.1f samples/sec; %.2f seconds)',
                        total_accuracy * 100, actual_bsize / duration, duration)
            logger.info('======================= Evaluation End =================================')
            logger.info('')


def main():
    apputil.initialize(variable_scope='tbcnn')
    if hyper.evaluation:
        do_evaluation()
    else:
        do_train()


if __name__ == '__main__':
    main()
