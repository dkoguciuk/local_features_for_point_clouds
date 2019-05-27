from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import importlib
import tensorflow as tf
from pathlib import Path

# Import PointCNN utils
import pointfly as pf

# Import config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from config import Config


def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


class PointCNN:
    def __init__(self, points, features, is_training, setting):

        # Params
        N = tf.shape(points)[0]
        xconv_params = setting.xconv_params
        sorting_method = setting.sorting_method
        with_X_transformation = setting.with_X_transformation

        # FPS sampling?
        if setting.sampling == 'fps':
            from sampling import tf_sampling

        # Let's go!!
        self.layer_pts = [points]
        if features is None:
            self.layer_fts = [features]
        else:
            features = tf.reshape(features, (N, -1, setting.data_dim - 3), name='features_reshape')
            C_fts = xconv_params[0]['C'] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']
            links = layer_param['links']
            if setting.sampling != 'random' and links:
                print('Error: flexible links are supported only when random sampling is used!')
                exit()

            # get k-nearest points
            pts = self.layer_pts[-1]
            fts = self.layer_fts[-1]
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
                qrs = self.layer_pts[-1]
            else:
                if setting.sampling == 'fps':
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                    qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (N, P, 3)
                elif setting.sampling == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                elif setting.sampling == 'random':
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
                else:
                    print('Unknown sampling method!')
                    exit()
            self.layer_pts.append(qrs)

            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (setting.with_global and layer_idx == len(xconv_params) - 1)
            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                              depth_multiplier, sorting_method, with_global)
            fts_list = []
            for link in links:
                fts_from_link = self.layer_fts[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                    fts_list.append(fts_slice)
            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_fts.append(fts_xconv)

        if hasattr(setting, 'xdconv_params'):
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

    def get_point_features(self):
        return self.layer_fts[1]


if __name__=='__main__':

    # Const
    MODEL_NAME = 'pointcnn_cls'
    SETTINGS_FILE = 'modelnet_settings'
    config = Config()

    # Import settings
    setting_path = os.path.join(str(Path().resolve()), MODEL_NAME)
    sys.path.append(setting_path)
    setting = importlib.import_module(SETTINGS_FILE)

    # List all settings
    sample_num = setting.sample_num
    rotation_range_val = setting.rotation_range_val
    scaling_range_val = setting.scaling_range_val
    jitter_val = setting.jitter_val

    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=1, name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Data placeholders
    shape = (config.batch_size, config.points_number, 6)
    data_val_placeholder = tf.placeholder(tf.float32, shape, name='data_val')
    label_val_placeholder = tf.placeholder(tf.int64, config.batch_size, name='label_val')
    handle = tf.placeholder(tf.string, shape=[], name='handle')

    # Iterator
    iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int64))
    (pts_fts, labels) = iterator.get_next()

    # Dataset
    dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
    if setting.map_fn is not None:
        dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
            setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
    dataset_val = dataset_val.batch(config.batch_size)
    batch_num_val = 1
    iterator_val = dataset_val.make_initializable_iterator()

    # Points/features
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    features_augmented = None
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled, [3, setting.data_dim - 3], axis=-1,
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

    # Net?
    point_features = PointCNN(points=points_augmented, features=features_augmented,
                              is_training=is_training, setting=config).get_point_features()
    print (point_features)