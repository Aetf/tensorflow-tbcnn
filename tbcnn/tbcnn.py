from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_fold as td


def main(word_dim=40):
    # Define the recursive block of 'process tree'
    expr_fwd = td.ForwardDeclaration(td.PyObjectType(), td.TensorType([word_dim, ]))

    word_embedding_layer = td.Embedding(
        len(word_embedding_model) + 1, word_dim, initializer=we_values, trainable=False)

    leaf_case = (td.InputTransform(lambda node: we_keys.get(node['w'].lower(), 0),
                                   name='leaf_input_transform')
                 >> td.Scalar('int32')
                 >> td.Function(word_embedding_layer, name='leaf_Function'))

    dep_embedding_layer = td.Embedding(len(dep_dict), param['dep_dim'], name='dep_embedding_layer')

    get_dep_embedding = (td.InputTransform(lambda d_label: dep_dict.get(d_label),
                                           name='dep_input_transform')
                         >> td.Scalar('int32')
                         >> td.Function(dep_embedding_layer, name='dep_embedding'))

    fclayer = td.FC(word_dim, name='process_tree_FC')
    non_leaf_case = (td.Record({'child': expr_fwd(), 'me': expr_fwd(), 'd': get_dep_embedding},
                               name='non-leaf_record')
                     >> td.Concat() >> td.Function(fclayer, name='non_leaf_function'))
    process_tree = td.OneOf(lambda node: node['is_leaf'], {
                            True: leaf_case, False: non_leaf_case}, name='process_tree_one_of')
    expr_fwd.resolve_to(process_tree)

    yblk = process_tree >> td.FC(len(y_classes))

    yorig = td.Vector(len(y_classes), name='yorig')

    # Compile the block
    compiler = td.Compiler.create((yblk, yorig))
    (y, y_) = compiler.output_tensors
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

    # train loop
    batch_size = 30
    train_set = compiler.build_loom_inputs(Input_train_tf)
    train_feed_dict = {}
    dev_feed_dict = compiler.build_feed_dict(Input_dev_tf)
    with tf.Session() as sess:
        sess.run(tf.global_variable_initializer())

        tf.summary.FileWriter('./tf_graph', graph=sess.graph)
        for epoch, shuffled in enumerate(td.epochs(train_set, epochs), 1):
            train_loss = 0.0
            for batch in td.group_by_batches(shuffled, batch_size):
                train_feed_dict[compiler.loom_input_tensor] = batch
                _, batch_loss = sess.run([train_step, cross_entropy], train_feed_dict)
                print(batch_loss)
                train_loss += np.sum(batch_loss)
            dev_loss = np.average(sess.run(cross_entropy, dev_feed_dict))
            print(dev_loss)


if __name__ == '__main__':
    main()
