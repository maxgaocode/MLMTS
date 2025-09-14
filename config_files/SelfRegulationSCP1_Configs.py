class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 6
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 114

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 32

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        '''
        add
        '''

        self.task_name = "classification"
        self.is_training = 1
        self.root_path = "./all_datasets/SelfRegulationSCP1/"
        self.data = "UEA"
        self.model_id = "SelfRegulationSCP1"
        self.selected_dataset = "SelfRegulationSCP1"

        # 模型配置
        self.model = "ModernTCN"
        self.ffn_ratio = 1
        self.patch_size = 1
        self.patch_stride = 1
        self.num_blocks = [1]
        self.large_size = [13]
        self.small_size = [5]
        self.dims = [32]

        # 训练配置
        self.head_dropout = 0.0
        self.dropout = 0.1
        self.class_dropout = 0.1
        self.itr = 1
        self.learning_rate = 0.001
        self.train_epochs = 100
        self.patience = 10
        self.training_mode = "self_supervised"


        self.stem_ratio = 6
        self.downsample_ratio = 2
        self.dw_dims = [256,256,256,256]
        self.enc_in = 6
        self.small_kernel_merged = 'False'
        self.use_multi_scale = 'False'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.freq = 'h'
        self.seq_len = 896
        self.individual = 0
        self.pred_len = 96
        self.num_class = 2
        self.decomposition = 0


        self.add = 32

'''
over
'''

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 4


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 1
