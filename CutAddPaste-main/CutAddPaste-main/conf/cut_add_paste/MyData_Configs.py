class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'MyData'
        # model configs
        self.input_channels = 80
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32
        self.project = 2

        self.dropout = 0.45
        # window_size 与 features_len 必须匹配：16->4, 32->6, 64->10
        self.features_len = 10
        self.window_size = 64
        self.time_step = 32

        # training configs
        self.num_epoch = 300

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4
        self.weight = 5e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 0.1
        # negative sample rates
        self.rate = 0.6
        # number of trend dimensions
        self.dim = 10
        # minimum cut length
        self.cut_rate = 12

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.005
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
