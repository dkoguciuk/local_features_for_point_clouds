import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../dgcnn'))
sys.path.append(os.path.join(BASE_DIR, '../dgcnn/utils'))
sys.path.append(os.path.join(BASE_DIR, '../dgcnn/models'))
import tf_util
from transform_nets import input_transform_net


def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl


def get_dgcnn_point_features(point_cloud, is_training, num_classes, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  k = 20

  # pairwise distance of the points in the point cloud
  adj_matrix = tf_util.pairwise_distance(point_cloud)

  # get indices of k nearest neighbors
  nn_idx = tf_util.knn(adj_matrix, k=k)

  # edge feature
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  # transform net 1
  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

  # point cloud transf
  point_cloud_transformed = tf.matmul(point_cloud, transform)

  # pairwise distance of the points in the point cloud
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)

  # get indices of k nearest neighbors
  nn_idx = tf_util.knn(adj_matrix, k=k)

  # I've got neighbors indices and subregion index (0-7)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  # Conv2d
  net = tf_util.conv2d(edge_feature,
                           64, [1,1], padding='VALID', stride=[1,1],
                           bn=True, is_training=is_training,
                           scope='dgcnn1', bn_decay=bn_decay)

  # Maxpool
  net = tf.reduce_max(net, axis=-2, keep_dims=False)
  return net


if __name__=='__main__':
  batch_size = 2
  num_pt = 1024
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed >= 0.5] = 1
  label_feed[label_feed < 0.5] = 0
  label_feed = label_feed.astype(np.int32)

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    point_features = get_dgcnn_point_features(input_pl, tf.constant(True), num_classes=40)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      point_features_eval = sess.run(point_features, feed_dict=feed_dict)
      print(point_features_eval.shape)
      print(point_features_eval)












