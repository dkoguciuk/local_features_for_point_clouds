import os
import importlib
import numpy as np
import tensorflow as tf
from config import Config
from flask import Flask, request, jsonify

###############################################################################
# PointNet-related
###############################################################################

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet', 'models'))

###############################################################################
# Consts
###############################################################################

# Model path
MODEL_PATH = 'pointnet/logs_modelnet/model_1/model_1.ckpt'

###############################################################################
# Flask app
###############################################################################

app = Flask(__name__)

###############################################################################
# Bulletproof
###############################################################################

# Check pointnet
if not os.path.exists('pointnet'):
    print()
    print('ERROR: You don\'t have pointnet cloned, please run:')
    print('git submodule update --init --recursive')
    print()
    exit(-1)

# Check model
if not os.path.exists(MODEL_PATH + '.index'):
    print()
    print('ERROR: You don\'t have pointnet model ckpt, please run:')
    print('HERE I SHOULD ADD STH LATER')
    print()
    exit(-1)

###############################################################################
# Load PointNet model
###############################################################################


def load_model(model):

    with tf.device('/gpu:0'):
        pointclouds_pl, _ = model.placeholder_inputs(Config.batch_size, Config.points_number)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        _, _, point_features = model.get_model(pointclouds_pl, is_training_pl, Config.classes_number)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    session = tf.Session(config=config)

    # Restore variables
    saver.restore(session, MODEL_PATH)

    # Return
    return session, pointclouds_pl, is_training_pl, point_features


# Load model
model = importlib.import_module('pointnet_cls')
session, pointclouds_pl, is_training_pl, point_features = load_model(model)

###############################################################################
# Flask API
###############################################################################


@app.route('/api', methods=['GET', 'POST'])
def pointnet_api():

    # Reconstruct point cloud
    data = request.json
    point_clouds = np.array(data['point_clouds'])

    # Inference
    point_features_eval = session.run(point_features,
                                      feed_dict={pointclouds_pl: point_clouds,
                                                 is_training_pl: False})
    point_features_eval = np.squeeze(point_features_eval)

    # Return
    return jsonify(features=point_features_eval.tolist())

###############################################################################
# Main
###############################################################################


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True)
