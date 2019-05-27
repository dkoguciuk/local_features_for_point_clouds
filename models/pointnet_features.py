import os
import sys
import tensorflow as tf

# Import config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from pointnet.utils import tf_util as pointnet_tf_util
from pointnet.models.transform_nets import input_transform_net as pointnet_input_transform_net


def get_model(point_cloud, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """

    with tf.variable_scope('transform_net1') as _:
        transform = pointnet_input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = pointnet_tf_util.conv2d(input_image, 64, [1,3],
                                  padding='VALID', stride=[1,1],
                                  bn=True, is_training=is_training,
                                  scope='conv1', bn_decay=bn_decay)
    net = pointnet_tf_util.conv2d(net, 64, [1,1],
                                  padding='VALID', stride=[1,1],
                                  bn=True, is_training=is_training,
                                  scope='conv2', bn_decay=bn_decay)

    return tf.squeeze(net, axis=-2)


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True), num_classes=40)
        print(outputs)
