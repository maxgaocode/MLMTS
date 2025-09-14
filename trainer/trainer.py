import os
import sys

sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import NTXentLoss
from sklearn.metrics import f1_score


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, lambda1, lambda2, lambda3, results1=None,
            results_F1=None):

    # 初始化结果列表（如果未提供）
    if results1 is None:
        results1 = []
    if results_F1 is None:
        results_F1 = []

    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    loss = []  # 用于存储每个轮次的损失组件

    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc, train_loss_details, train_f1 = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
            criterion, train_dl, config, device, training_mode, lambda1, lambda2, lambda3)

        valid_loss, valid_acc, valid_f1, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode  )

        # 学习率调度
        if training_mode != 'self_supervised':
            scheduler.step(valid_loss)

        # 收集损失组件
        loss.append(train_loss_details)

        # 输出训练与验证信息
        # logger.debug(f'\nEpoch : {epoch}\n'
        #              f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\t | \tTrain F1 Score     : {train_f1:2.4f}\n'
        #              f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\t | \tValid F1 Score     : {valid_f1:2.4f}')

    # 保存模型
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    # 保存损失历史
    os.makedirs(os.path.join(experiment_log_dir, "saved_loss"), exist_ok=True)
    np.save(os.path.join(experiment_log_dir, "saved_loss", 'loss_test.npy'), loss)

    # 在非自监督模式下评估最终测试集性能
    if training_mode != "self_supervised":
        # evaluate on the test set
        
        test_loss, test_acc, test_f1, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\t | Test F1 Score      : {test_f1:0.4f}')

        # 收集并打印多次实验结果
        results1.append(test_acc)
        results_F1.append(test_f1)
        print(f"Accuracies: {results1}\n, F1 Scores: {results_F1}")

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode,lambda1,lambda2, lambda3):
    """
    修改后的模型训练函数，添加了损失组件跟踪与F1分数计算
    """
    total_loss = []
    total_acc = []
    all_preds = []  # 用于计算F1分数
    all_labels = []  # 用于计算F1分数

    # 初始化损失组件列表
    loss_TC_1 = []
    loss_TC_2 = []
    loss_NT_xent = []
    loss_placeholder = []
 

    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1, x_k1 = model(aug1)
            predictions2, features2, x_k2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_view_loss1,temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2, x_k1)
            temp_view_loss2,temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1, x_k2)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2

        else:
            # 处理不同的返回值格式
            output = model(data)
            if isinstance(output, tuple) and len(output) == 3:
                predictions, features, _ = output  # 如果模型返回3个值，忽略第3个
            else:
                predictions, features = output  # 原来的方式

        # compute loss
        if training_mode == "self_supervised":

            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            nt_xent_loss = nt_xent_criterion(zis, zjs)

            # 计算总损失，与GNN版本保持一致
            loss1= (temp_cont_loss1 + temp_cont_loss2) * lambda1 
            loss2=(temp_view_loss1+temp_view_loss2)* lambda2
            loss=loss1+loss2+nt_xent_loss* lambda3
            print('cross temporal loss:{},  and  view temporal loss:{},   and nt_xent_LOSS :{}'.format(loss1,loss2,nt_xent_loss* lambda2)) 

            # 记录各组件损失
            loss_TC_1.append(temp_cont_loss1.item())
            loss_TC_2.append(temp_view_loss1.item())
            loss_NT_xent.append(nt_xent_loss.item())

        else:  # supervised training or fine tuning

            loss = criterion(predictions, labels)            
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            # 收集预测和标签用于计算F1分数
            all_preds.extend(predictions.detach().argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        total_loss.append(loss.item())       
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
        total_f1 = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
        # 计算F1分数
        total_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0

    # 返回损失组件，与trainer_GNN.py保持一致
    return total_loss, total_acc, [
        np.mean(loss_TC_1) if loss_TC_1 else 0,
        np.mean(loss_TC_2) if loss_TC_2 else 0,
        np.mean(loss_NT_xent) if loss_NT_xent else 0,
        np.mean(loss_placeholder) if loss_placeholder else 0
    ], total_f1


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    """
    修改后的模型评估函数，添加了F1分数计算
    """
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []
    all_preds = []  # 用于计算F1分数
    all_labels = []  # 用于计算F1分数

    criterion = nn.CrossEntropyLoss()
    outs = []
    trgs = []

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)
                # 处理不同的返回值格式
                if isinstance(output, tuple) and len(output) == 3:
                    predictions, features, _ = output  # 如果模型返回3个值，忽略第3个
                else:
                    predictions, features = output  # 原来的方式

            # compute loss
            if training_mode != "self_supervised":
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                # 收集预测和标签用于计算F1分数
                all_preds.extend(predictions.detach().argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs.extend(pred.cpu().numpy().flatten())
                trgs.extend(labels.data.cpu().numpy().flatten())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0

    if training_mode == "self_supervised":
        total_acc = 0
        total_f1 = 0
        return total_loss, total_acc, total_f1, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
        # 计算F1分数
        total_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0

    return total_loss, total_acc, total_f1, outs, trgs