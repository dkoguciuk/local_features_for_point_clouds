import os

class Config:

    def __init__(self):

        self.batch_size = 32
        self.points_number = 1024
        self.classes_number = 40
        self.server_host = os.getenv('SOCKET_SERVER', 'http://localhost:3000')

        ###########################################################################
        # PointCNN feature extractor settings
        ###########################################################################

        # xconv
        self.sorting_method = None
        self.with_X_transformation = True
        self.xconv_param_name = ('K', 'D', 'P', 'C', 'links')
        self.xconv_params = [dict(zip(self.xconv_param_name, xconv_param)) for xconv_param in [(8, 1, -1, 48, [])]]

        # Sampling
        self.sampling = 'random'
        self.with_global = False

        ###########################################################################
        # Local features extractors training
        ###########################################################################

        # load_fn = data_utils.load_cls_train_val
        # balance_fn = None
        # map_fn = None
        # keep_remainder = True
        # save_ply_fn = None
        #
        # num_class = 40
        # sample_num = 1024
        # batch_size = 128
        #
        # num_epochs = 1024
        # step_val = 500
        #
        # learning_rate_base = 0.01
        # decay_steps = 8000
        # decay_rate = 0.5
        # learning_rate_min = 1e-6
        #
        # weight_decay = 1e-5
        #
        # jitter = 0.0
        # jitter_val = 0.0
        #
        # rotation_range = [0, math.pi, 0, 'u']
        # rotation_range_val = [0, 0, 0, 'u']
        # rotation_order = 'rxyz'
        #
        # scaling_range = [0.1, 0.1, 0.1, 'g']
        # scaling_range_val = [0, 0, 0, 'u']
        #
        # sample_num_variance = 1 // 8
        # sample_num_clip = 1 // 4
        #
        # optimizer = 'adam'
        # epsilon = 1e-2
        #
        # data_dim = 6
        # use_extra_features = False


