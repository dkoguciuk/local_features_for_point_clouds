import os
import importlib
import numpy as np
import tensorflow as tf
from pathlib import Path
from config import Config
from flask import Flask, request, jsonify

###############################################################################
# PointCNN-related
###############################################################################

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointcnn'))
import pointfly as pf

###############################################################################
# Consts
###############################################################################

# Model path
MODEL_NAME = 'pointcnn_cls'
SETTINGS_FILE = 'modelnet_x3_l4'
MODEL_PATH = 'pointcnn/logs_modelnet/model_1/model_1'
sys.path.append(os.path.join(BASE_DIR, 'pointcnn', MODEL_NAME))

###############################################################################
# Flask app
###############################################################################

app = Flask(__name__)

###############################################################################
# Bulletproof
###############################################################################

# Check pointnet
if not os.path.exists('pointcnn'):
    print()
    print('ERROR: You don\'t have pointcnn cloned, please run:')
    print('git submodule update --init --recursive')
    print()
    exit(-1)

# Check model
if not os.path.exists(MODEL_PATH + '.index'):
    print()
    print('ERROR: You don\'t have pointcnn model ckpt, please run:')
    print('HERE I SHOULD ADD STH LATER')
    print()
    exit(-1)

###############################################################################
# PLACEHOLDERS
###############################################################################

# Import model
model = importlib.import_module(MODEL_NAME)

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
shape = (Config.batch_size, Config.points_number, 6)
data_val_placeholder = tf.placeholder(tf.float32, shape, name='data_val')
label_val_placeholder = tf.placeholder(tf.int64, Config.batch_size, name='label_val')
handle = tf.placeholder(tf.string, shape=[], name='handle')

# Iterator
iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.int64))
(pts_fts, labels) = iterator.get_next()

# Dataset
dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
if setting.map_fn is not None:
    dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
        setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
dataset_val = dataset_val.batch(Config.batch_size)
batch_num_val = 1
iterator_val = dataset_val.make_initializable_iterator()

# Points/features
pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
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

###############################################################################
# MODEL DEFINITION
###############################################################################

net = model.Net(points=points_augmented, features=features_augmented, is_training=is_training, setting=setting)
point_features = net.point_features

###############################################################################
# Create session
###############################################################################

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
session = tf.Session(config=config)

# Load the model
saver = tf.train.Saver()
saver.restore(session, MODEL_PATH)

# Data handle
handle_val = session.run(iterator_val.string_handle())

# Get the xforms and rotations
xforms_np, rotations_np = pf.get_xforms(Config.batch_size, rotation_range=rotation_range_val,
                                        scaling_range=scaling_range_val, order=setting.rotation_order)

###############################################################################
# Flask API
###############################################################################


@app.route('/api', methods=['GET', 'POST'])
def pointcnn_api():

    # Reconstruct point cloud
    data = request.json
    point_clouds = np.array(data['point_clouds'])

    # Feed dataset iterator
    blind_labels = np.zeros(point_clouds.shape[0], dtype=np.int64)
    session.run(iterator_val.initializer,
                feed_dict={data_val_placeholder: point_clouds,
                           label_val_placeholder: blind_labels})

    # Inference
    point_features_eval = session.run(point_features,
                                      feed_dict={handle: handle_val,
                                                 indices: pf.get_indices(Config.batch_size, sample_num, Config.points_number),
                                                 xforms: xforms_np, rotations: rotations_np,
                                                 jitter_range: np.array([jitter_val]), is_training: False})


    # Return
    return jsonify(features=point_features_eval.tolist())

###############################################################################
# Main
###############################################################################


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(port=port, debug=True)
