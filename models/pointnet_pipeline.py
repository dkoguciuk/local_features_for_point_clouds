import tf_util
import tensorflow as tf


def placeholder_inputs(batch_size, num_point, num_features):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


def get_model(point_features, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_features.get_shape()[0].value
    num_point = point_features.get_shape()[1].value
    num_features = point_features.get_shape()[2].value
    end_points = {}
    input_image = tf.expand_dims(point_features, -1)
    
    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 128, [1, num_features],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    point_features = net

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

    # Return
    return net, end_points, point_features


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True), num_classes=40)
        print(outputs)
