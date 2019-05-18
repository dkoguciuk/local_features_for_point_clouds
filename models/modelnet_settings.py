#!/usr/bin/python3

import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../pointcnn'))
import data_utils


load_fn = data_utils.load_cls_train_val

###############################################################################
# Batch size
###############################################################################

batch_size = 4

###############################################################################
# ModelNet data is shipped with 2048 points, how many points do we want to use?
###############################################################################

point_num = 1024

###############################################################################
# ModelNet data is shipped with 2048 points, how should we sample it to get
# num_points?:
#   random - take desired number of points from 2048 points in dataset randomly
#   fps - take desired number of points from 2048 points in dataset using fps
###############################################################################

dataset_sampling = 'fps'


###############################################################################
# How many dimensions of the input data should we use?
###############################################################################

data_dim = 6

###############################################################################
# Data augmentation ranges
###############################################################################

jitter = 0.0
rotation_range = [0, math.pi, 0, 'u']
rotation_order = 'rxyz'
scaling_range = [0.1, 0.1, 0.1, 'g']

jitter_val = 0.0
rotation_range_val = [0, 0, 0, 'u']
scaling_range_val = [0, 0, 0, 'u']

###############################################################################
# Learning process
###############################################################################

learning_rate_base = 0.001
decay_steps = 8000
decay_rate = 0.5
learning_rate_min = 1e-5
optimizer = 'adam'
epsilon = 1e-2

point_num_variance = 1 // 8
point_num_clip = 1 // 4
validate_after_batches = 100





balance_fn = None
map_fn = None
keep_remainder = True
save_ply_fn = None

num_class = 40

num_epochs = 250

















x = 3

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 2, 384, 32 * x, []),
                 (16, 2, 128, 64 * x, []),
                 (16, 3, 128, 128 * x, [])]]

with_global = True

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(128 * x, 0.0),
              (64 * x, 0.8)]]

sampling = 'random'




use_extra_features = False
with_X_transformation = True
sorting_method = None
