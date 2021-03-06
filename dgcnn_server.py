import os
import importlib
import numpy as np
import tensorflow as tf
from config import Config
from flask import Flask, request, jsonify

###############################################################################
# DGCNN-related
###############################################################################

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'dgcnn', 'models'))

###############################################################################
# Consts
###############################################################################

# Model path
MODEL_PATH = 'dgcnn/logs_modelnet/model_1/model_1.ckpt'
config = Config()

###############################################################################
# Flask app
###############################################################################

app = Flask(__name__)

###############################################################################
# Bulletproof
###############################################################################

# Check pointnet
if not os.path.exists('dgcnn'):
    print()
    print('ERROR: You don\'t have dgcnn cloned, please run:')
    print('git submodule update --init --recursive')
    print()
    exit(-1)

# Check model
if not os.path.exists(MODEL_PATH + '.index'):
    print()
    print('ERROR: You don\'t have dgcnn model ckpt, please run:')
    print('HERE I SHOULD ADD STH LATER')
    print()
    exit(-1)

###############################################################################
# Load DGCNN model
###############################################################################


def load_model(model):

    with tf.device('/gpu:0'):
        pointclouds_pl, _ = model.placeholder_inputs(config.batch_size, config.points_number)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        _, _, point_features = model.get_model(pointclouds_pl, is_training_pl, config.classes_number)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = True
    session = tf.Session(config=session_config)

    # Restore variables
    saver.restore(session, MODEL_PATH)

    # Return
    return session, pointclouds_pl, is_training_pl, point_features


# Load model
model = importlib.import_module('dgcnn')
session, pointclouds_pl, is_training_pl, point_features = load_model(model)

###############################################################################
# Flask API
###############################################################################


@app.route('/api', methods=['GET', 'POST'])
def dgcnn_api():

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
    port = int(os.environ.get('PORT', 5001))
    app.run(port=port, debug=True)
