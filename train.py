import torch
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad, str2bool
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
import shutil

# Args selections
start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Run Description')
parser.add_argument('--seed', default=123, type=int, help='seed value')
parser.add_argument('--training_mode', default='self_supervised', type=str, help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear, both')
parser.add_argument('--selected_dataset', default='FingerMovements', type=str, help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,  help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,help='Project home directory')
parser.add_argument('--wd', default=4e-5, type=float, help='wi ')
parser.add_argument('--lambda1', default=0.5, type=float, help='hyperpamater')
parser.add_argument('--lambda2', default=0.5, type=float, help='hyperpamater')
parser.add_argument('--lambda3', default=0.5, type=float, help='hyperpamater')

##maxgao
parser.add_argument('--ffn_ratio', default=2, type=int, help='ffn_ratio ')
parser.add_argument('--stem_ratio', default=6, type=int, help='stem_ratio ')
parser.add_argument('--downsample_ratio', default=2, type=int, help='downsample_ratio ')

parser.add_argument('--dropout', default=0.1, type=float, help='dropout ')
parser.add_argument('--head_dropout', default=0.5, type=float, help='head_dropout ')

args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
original_training_mode = args.training_mode  # 保存原始训练模式
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# 修复随机种子
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



# 修改函数定义，添加额外的参数
def run_training(current_mode, args, current_seed=None, results_acc=None, results_f1=None):
    # 使用传入的种子替代全局种子
    if current_seed is None:
        current_seed = SEED  # 使用默认种子

    # 设置随机种子
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)

    # 设置日志目录
    current_experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                              current_mode + f"_seed_{current_seed}")
    os.makedirs(current_experiment_log_dir, exist_ok=True)

    # 设置日志
    log_file_name = os.path.join(current_experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    current_logger = _logger(log_file_name)
    current_logger.debug("=" * 45)
    current_logger.debug(f'Dataset: {data_type}')
    current_logger.debug(f'Method:  {method}')
    current_logger.debug(f'Mode:    {current_mode}')
    current_logger.debug("=" * 45)

    # 加载数据集（每次重新加载以确保一致性）
    train_dl, valid_dl, test_dl = data_generator(data_path, configs, current_mode)
    current_logger.debug("Data loaded ...")

    # 加载模型
    model = base_Model(configs,args).to(device)
    temporal_contr_model = TC(configs,args, device).to(device)

    # 根据训练模式设置不同的模型加载逻辑
    if current_mode == "train_linear":
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{current_seed}",
                         "saved_models"))
        try:
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()

            # 过滤掉logits层的参数
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]

            # 只加载形状匹配的参数
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_dict[k] = v

            current_logger.debug(f"Loading {len(compatible_dict)}/{len(model_dict)} compatible parameters")
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)
        except Exception as e:
            current_logger.error(f"Error loading pre-trained model: {e}")
            current_logger.error("Training from scratch instead...")

    # 设置优化器
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                betas=(configs.beta1, configs.beta2), weight_decay=args.wd)

    # 如果是自监督模式，复制相关文件
    if current_mode == "self_supervised":
        copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

    # 训练模型
    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer,
            train_dl, valid_dl, test_dl, device, current_logger, configs,
            current_experiment_log_dir, current_mode,args.lambda1,args.lambda2,args.lambda3)

    # 如果不是自监督模式，进行测试评估
    if current_mode != "self_supervised":
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, current_mode)
        total_loss, total_acc, total_f1, pred_labels, true_labels = outs
        _calc_metrics(pred_labels, true_labels, current_experiment_log_dir, args.home_path)

        # 如果结果列表存在，则添加结果
        if results_acc is not None:
            results_acc.append(total_acc)
        if results_f1 is not None:
            results_f1.append(total_f1)

    return model, temporal_contr_model


# 主程序开始
data_path = f"./data/{data_type}"

results_accuracy = []
results_f1 = []
num_trials =7

print(f"Starting {num_trials} trials for {data_type} dataset")
print(f"Training mode: {original_training_mode}")
print("=" * 50)

seeds=[ 77,124,65,132,44,32, 653]
# 多次运行循环
for trial in range(num_trials):
    print(f"Trial {trial + 1}/{num_trials}")

    # 为每次运行设置不同的种子
    current_seed = seeds[trial]

    args.original_training_mode = "self_supervised"
        #print(f"  Phase 1: Running self-supervised learning (Seed: {current_seed})...")
        #run_training("self_supervised",args, current_seed, [], [])  # 自监督阶段不收集结果

    print(f"  Phase 2: Running fine-tuning (Seed: {current_seed})...")
    _, _ = run_training("self_supervised", args, current_seed, results_accuracy, results_f1)

    args.original_training_mode = "training_line"        # 运行单个指定的训练模式
    _, _ = run_training('training_line', args, current_seed, results_accuracy, results_f1)

    # 显示当前进度和结果
    if results_accuracy:
        current_avg_acc = sum(results_accuracy) / len(results_accuracy)
        current_avg_f1 = sum(results_f1) / len(results_f1)
        print(f"  Current results - Trial {trial + 1}")
        print(f"  Accuracy: {results_accuracy[-1]:.4f}, F1-Score: {results_f1[-1]:.4f}")
        print(f"  Running average - Accuracy: {current_avg_acc:.4f}, F1-Score: {current_avg_f1:.4f}")

    print("-" * 50)

# 计算并显示总体结果
if results_accuracy and results_f1:
    total_acc = sum(results_accuracy)
    total_f1 = sum(results_f1)
    avg_acc = total_acc / len(results_accuracy)
    avg_f1 = total_f1 / len(results_f1)
    std_acc = np.std(results_accuracy)
    std_f1 = np.std(results_f1)

    print("\n" + "=" * 50)
    print(f"Final Results Summary:")
    print(f"Dataset: {data_type}, Mode: {original_training_mode}, Trials: {num_trials}")
    print(f"Total accuracy: {total_acc:.4f}")
    print(f"Average accuracy: {avg_acc:.4f} (±{std_acc:.4f})")
    print(f"Total F1 score: {total_f1:.4f}")
    print(f"Average F1 score: {avg_f1:.4f} (±{std_f1:.4f})")
    print("=" * 50)

    # 保存详细结果到文件
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    seeds=[ 77,124,65,132,44,32, 653]

    f = open("Abystudy_{}.csv".format(original_training_mode), 'a')  #Abystudy_self_supervised
    #f = open("ConTraResults_{}.csv".format(data_type), 'a')
    f.write("Dataset:{},lambda1:{},lambda2:{},lambda3:{},trials:{}, ".format(data_type,args.lambda1,args.lambda2,args.lambda3,num_trials))
    

    f.write("Individual trial results:")
    for i, (acc, f1) in enumerate(zip(results_accuracy, results_f1)):
        args.seed=seeds[i]

        f.write(f"Trial {i + 1} (Seed {args.seed}) - Accuracy: {acc:.4f}, F1-Score: {f1:.4f},")
    f.write("\n")
    f.close()




print(f"Total execution time: {datetime.now() - start_time}")

