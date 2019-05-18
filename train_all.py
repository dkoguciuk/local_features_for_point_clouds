#!/usr/bin/python3
"""Training and Validation On Classification Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


import pointfly as pf
import dgcnn_features as dgcnn_features_module
import pointnet_pipeline as pointnet_pipeline_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='modelnet',
                        help='Dataset to train on: modelnet or shapenet [default: modelnet]')
    parser.add_argument('--save_folder', '-s',  default='log',
                        help='Path to folder for saving check points and summary')
    parser.add_argument('--setting', '-x', default='modelnet_settings', help='Setting to use')
    parser.add_argument('--model', '-m', default='pointcnn_features', help='Model to use')
    parser.add_argument('--epochs', help='Number of training epochs (default defined in setting)', type=int)
    parser.add_argument('--batch_size', help='Batch size (default defined in setting)', type=int)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%s_%d' % (args.model, args.setting, time_string, os.getpid()))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = args.epochs or setting.num_epochs
    batch_size = args.batch_size or setting.batch_size
    data_dim = setting.data_dim
    point_num = setting.point_num
    validate_after_batches = setting.validate_after_batches
    rotation_range = setting.rotation_range
    rotation_range_val = setting.rotation_range_val
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val
    pool_setting_val = None if not hasattr(setting, 'pool_setting_val') else setting.pool_setting_val
    pool_setting_train = None if not hasattr(setting, 'pool_setting_train') else setting.pool_setting_train
    dataset_sampling = setting.dataset_sampling

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data_train, label_train, data_val, label_val = setting.load_fn(
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'),
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    if setting.balance_fn is not None:
        num_train_before_balance = data_train.shape[0]
        repeat_num = setting.balance_fn(label_train)
        data_train = np.repeat(data_train, repeat_num, axis=0)
        label_train = np.repeat(label_train, repeat_num, axis=0)
        data_train, label_train = data_utils.grouped_shuffle([data_train, label_train])
        num_epochs = math.floor(num_epochs * (num_train_before_balance / data_train.shape[0]))

    if setting.save_ply_fn is not None:
        folder = os.path.join(root_folder, 'pts')
        print('{}-Saving samples as .ply files to {}...'.format(datetime.now(), folder))
        sample_num_for_ply = min(512, data_train.shape[0])
        if setting.map_fn is None:
            data_sample = data_train[:sample_num_for_ply]
        else:
            data_sample_list = []
            for idx in range(sample_num_for_ply):
                data_sample_list.append(setting.map_fn(data_train[idx], 0)[0])
            data_sample = np.stack(data_sample_list)
        setting.save_ply_fn(data_sample, folder)

    num_train = data_train.shape[0]
    dataset_point_num = data_train.shape[1]
    dataset_feature_num = data_train.shape[2]
    num_val = data_val.shape[0]
    print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, point_num, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(batch_size, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(batch_size, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    data_train_placeholder = tf.placeholder(data_train.dtype, data_train.shape, name='data_train')
    label_train_placeholder = tf.placeholder(tf.int64, label_train.shape, name='label_train')
    data_val_placeholder = tf.placeholder(data_val.dtype, data_val.shape, name='data_val')
    label_val_placeholder = tf.placeholder(tf.int64, label_val.shape, name='label_val')
    handle = tf.placeholder(tf.string, shape=[], name='handle')

    ######################################################################
    dataset_train = tf.data.Dataset.from_tensor_slices((data_train_placeholder, label_train_placeholder))
    dataset_train = dataset_train.shuffle(buffer_size=batch_size * 4)

    if setting.map_fn is not None:
        dataset_train = dataset_train.map(lambda data, label:
                                          tuple(tf.py_func(setting.map_fn, [data, label], [tf.float32, label.dtype])),
                                          num_parallel_calls=setting.num_parallel_calls)

    if setting.keep_remainder:
        dataset_train = dataset_train.batch(batch_size)
        batch_num_per_epoch = math.ceil(num_train / batch_size)
    else:
        dataset_train = dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        batch_num_per_epoch = math.floor(num_train / batch_size)
    dataset_train = dataset_train.repeat(num_epochs)
    iterator_train = dataset_train.make_initializable_iterator()
    batch_num = batch_num_per_epoch * num_epochs
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))

    dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
    if setting.map_fn is not None:
        dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
            setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
    if setting.keep_remainder:
        dataset_val = dataset_val.batch(batch_size)
        batch_num_val = math.ceil(num_val / batch_size)
    else:
        dataset_val = dataset_val.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        batch_num_val = math.floor(num_val / batch_size)
    iterator_val = dataset_val.make_initializable_iterator()
    print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

    ###########################################################################
    # Build data iterator
    ###########################################################################

    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types,
                                                   output_shapes=((batch_size, dataset_point_num, data_dim),
                                                                  (batch_size)))
    (pts_fts, labels) = iterator.get_next()

    ###########################################################################
    # Sample points from dataset
    ###########################################################################

    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')

    ###########################################################################
    # Augment points and features
    ###########################################################################

    features_augmented = None
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    features_augmented = pf.augment(features_sampled, rotations)
                else:
                    normals, rest = tf.split(features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations)
                    features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                features_augmented = features_sampled
    else:
        points_sampled = pts_fts_sampled
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    ###########################################################################
    # Build features
    ###########################################################################

    pointcnn = model.PointCNN(points=points_augmented, features=features_augmented,
                              is_training=is_training, setting=setting)
    pointcnn_features = pointcnn.get_point_features()

    dgcnn_features = dgcnn_features_module.get_dgcnn_point_features(point_cloud=points_augmented,
                                                                    is_training=is_training,
                                                                    num_classes=setting.num_class,
                                                                    bn_decay=None)

    ###########################################################################
    # Build pipeline on top of it
    ###########################################################################

    # PointNet pipeline
    point_features = tf.concat((pointcnn_features, dgcnn_features), axis=-1)
    predictions, _, _ = pointnet_pipeline_module.get_model(point_features=point_features, is_training=is_training,
                                                           num_classes=setting.num_class, bn_decay=None)

    # Training op
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    loss = tf.reduce_mean(loss)
    loss = pointnet_pipeline_module.get_loss(predictions, labels)
    tf.summary.scalar('loss/train', tensor=loss, collections=['train'])
    tf.summary.scalar('loss/test', tensor=loss, collections=['test'])

    # Summaries
    correct = tf.equal(tf.argmax(predictions, 1), tf.to_int64(labels))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batch_size)
    tf.summary.scalar('accuracy/train', tensor=accuracy, collections=['train'])
    tf.summary.scalar('accuracy/test', tensor=accuracy, collections=['test'])

    # Reset metrics
    reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

    # Learning rate decay
    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
    # Learning rate clip
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])

    # Optimizer
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)

    # Train op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    # Init op
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)

    # backup all code
    #code_folder = os.path.abspath(os.path.dirname(__file__))
    #shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)))

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    # Parameter num
    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    # Session!
    with tf.Session() as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_test_op = tf.summary.merge_all('test')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        # Init!
        sess.run(init_op)

        # # Load the model
        # if args.load_ckpt is not None:
        #     saver.restore(sess, args.load_ckpt)
        #     print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        # Data
        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())
        sess.run(iterator_train.initializer, feed_dict={
            data_train_placeholder: data_train,
            label_train_placeholder: label_train,
        })

        for batch_idx_train in range(batch_num):

            ###################################################################
            # Training
            ###################################################################

            if not setting.keep_remainder \
                    or num_train % batch_size == 0 \
                    or (batch_idx_train % batch_num_per_epoch) != (batch_num_per_epoch - 1):
                batch_size_train = batch_size
            else:
                batch_size_train = num_train % batch_size

            # Transforms
            offset = int(random.gauss(0, point_num * setting.point_num_variance))
            offset = max(offset, -point_num * setting.point_num_clip)
            offset = min(offset, point_num * setting.point_num_clip)
            point_num_train = point_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size_train,
                                                    rotation_range=rotation_range,
                                                    scaling_range=scaling_range,
                                                    order=setting.rotation_order)

            # Run!
            sess.run(reset_metrics_op)
            _, loss_train, acc_train, summaries_train = sess.run([train_op, loss, accuracy, summaries_op],
                     feed_dict={
                         handle: handle_train,
                         indices: pf.get_indices(batch_size_train, point_num_train, dataset_point_num,
                                                 dataset_sampling, pool_setting_train),
                         xforms: xforms_np,
                         rotations: rotations_np,
                         jitter_range: np.array([jitter]),
                         is_training: True,
                     })
            if batch_idx_train % 10 == 0:
                summary_writer.add_summary(summaries_train, batch_idx_train)
                print('{}-[Train]-Iter: {:06d}  Loss: {:.4f}  Acc: {:.4f}'
                      .format(datetime.now(), batch_idx_train, loss_train, acc_train))
                sys.stdout.flush()

            ###################################################################
            # Validation
            ###################################################################

            #if (batch_idx_train % validate_after_batches == 0 and (batch_idx_train != 0 or args.load_ckpt is not None)) or batch_idx_train == batch_num - 1:
            if (batch_idx_train % validate_after_batches == 0 and (batch_idx_train != 0)) or batch_idx_train == batch_num - 1:
                sess.run(iterator_val.initializer, feed_dict={
                    data_val_placeholder: data_val,
                    label_val_placeholder: label_val,
                })
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                sess.run(reset_metrics_op)
                for batch_idx_val in range(batch_num_val):
                    if not setting.keep_remainder \
                            or num_val % batch_size == 0 \
                            or batch_idx_val != batch_num_val - 1:
                        batch_size_val = batch_size
                    else:
                        batch_size_val = num_val % batch_size
                    xforms_np, rotations_np = pf.get_xforms(batch_size_val,
                                                            rotation_range=rotation_range_val,
                                                            scaling_range=scaling_range_val,
                                                            order=setting.rotation_order)
                    loss_val, acc_val, summaries_val = sess.run([loss, accuracy, summaries_op],
                             feed_dict={
                                 handle: handle_val,
                                 indices: pf.get_indices(batch_size_val, point_num, dataset_point_num,
                                                         dataset_sampling),
                                 xforms: xforms_np,
                                 rotations: rotations_np,
                                 jitter_range: np.array([jitter_val]),
                                 is_training: False,
                             })
                summary_writer.add_summary(summaries_val, batch_idx_train)
                print('{}-[Val  ]-Average:      Loss: {:.4f}  Acc: {:.4f}'.format(datetime.now(), loss_val, acc_val))
                sys.stdout.flush()
            ######################################################################

        print('{}-Done!'.format(datetime.now()))

if __name__ == '__main__':
    main()
