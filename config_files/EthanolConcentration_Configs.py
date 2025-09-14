class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 3
        self.kernel_size = 4
        self.stride = 1
        self.final_out_channels = 128
        self.num_classes = 4
        self.dropout = 0.35

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 8

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        '''
         add
         '''

        self.task_name = "classification"
        self.is_training = 1
        self.root_path = "./all_datasets/EthanolConcentration/"
        self.data = "UEA"
        self.model_id = "EthanolConcentration"
        self.selected_dataset = "EthanolConcentration"

        # 模型配置
        self.model = "ModernTCN"
        self.ffn_ratio = 1
        self.patch_size = 4
        self.patch_stride = 1
        self.num_blocks = [1]
        self.large_size = [21]
        self.small_size = [7]
        self.dims = [128]

        # 训练配置
        self.head_dropout = 0.0
        self.dropout = 0.1
        self.class_dropout = 0.1
        self.itr = 1
        self.learning_rate = 3e-4
        self.train_epochs = 100
        self.patience = 10
        self.training_mode = "self_supervised"

        self.stem_ratio = 6
        self.downsample_ratio = 2
        self.dw_dims = [256, 256, 256, 256]
        self.enc_in = 3
        self.small_kernel_merged = 'False'
        self.use_multi_scale = 'False'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.freq = 'h'
        self.seq_len = 1750
        self.individual = 0
        self.pred_len = 96
        self.num_class = 4
        self.decomposition = 0

        self.add = 128
'''
over
'''
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 2


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 1
