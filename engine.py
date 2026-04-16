import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from tqdm import tqdm
import json
import os
import numpy as np


import torch.nn.functional as F

# ==================== DG 核心模块 ====================
def physical_augmentation(x, num_packets=5, mask_prob=0.3, shuffle_prob=0.3):
    """
    物理感知张量增强 (运行于 GPU，无额外 IO 开销)
    x: [B, 1, L] 的流量张量
    """
    B, C, L = x.shape
    packet_size = L // num_packets
    x_phy = x.clone()
    
    for i in range(B):
        rand_val = torch.rand(1).item()
        # 1. 丢包模拟 (Mask)
        if rand_val < mask_prob:
            mask_idx = torch.randint(0, num_packets, (1,)).item()
            x_phy[i, :, mask_idx*packet_size : (mask_idx+1)*packet_size] = 0.0
            
        # 2. 乱序/重传模拟 (Shuffle & Duplicate)
        elif rand_val < (mask_prob + shuffle_prob):
            idx1 = torch.randint(0, num_packets-1, (1,)).item()
            idx2 = idx1 + 1
            if torch.rand(1).item() < 0.5: # 交换
                temp = x_phy[i, :, idx1*packet_size : (idx1+1)*packet_size].clone()
                x_phy[i, :, idx1*packet_size : (idx1+1)*packet_size] = x_phy[i, :, idx2*packet_size : (idx2+1)*packet_size]
                x_phy[i, :, idx2*packet_size : (idx2+1)*packet_size] = temp
            else: # 重传 (复制覆盖)
                x_phy[i, :, idx2*packet_size : (idx2+1)*packet_size] = x_phy[i, :, idx1*packet_size : (idx1+1)*packet_size]
    return x_phy

def statistical_augmentation(z, alpha=0.1):
    """
    统计感知特征增强
    z: [B, D] 原始的高维特征
    """
    mu = z.mean(dim=-1, keepdim=True)
    sigma = z.std(dim=-1, keepdim=True)
    
    # 随机漂移因子
    gamma = torch.randn_like(sigma) * alpha + 1.0
    beta = torch.randn_like(mu) * alpha
    
    # Instance Norm + 重设风格
    z_stat = (z - mu) / (sigma + 1e-6) * (sigma * gamma) + (mu + beta)
    return z_stat

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()

class StyleDivergencePushLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_sty_x, z_sty_xstat):
        """
        最小化原始视图与统计漂移视图在风格空间的余弦相似度。
        通过将其作为损失进行优化（下降），强迫两个特征向量在风格空间中指向相反方向，
        从而激活风格探测器捕捉环境扰动。
        """
        # 计算余弦相似度，范围在 [-1, 1]。优化使其趋近 -1（相互排斥）
        cos_sim = F.cosine_similarity(z_sty_x, z_sty_xstat, dim=-1)
        return torch.mean(cos_sim)


class OrthogonalIsolationPenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_con, z_sty):
        """
        最小化内容特征与风格特征之间互相关矩阵的 Frobenius 范数。
        强制两子空间绝对统计独立与正交。
        """
        # 1. 均值中心化
        z_con = z_con - z_con.mean(dim=0, keepdim=True)
        z_sty = z_sty - z_sty.mean(dim=0, keepdim=True)
        
        # 2. L2 归一化
        z_con = F.normalize(z_con, p=2, dim=0)
        z_sty = F.normalize(z_sty, p=2, dim=0)
        
        # 3. 计算互相关矩阵 (Batch 维度相乘)
        correlation_matrix = torch.matmul(z_con.T, z_sty)
        
        # 4. 计算 Frobenius 范数的平方
        return torch.norm(correlation_matrix, p='fro') ** 2


def train_one_epoch_dg(model: torch.nn.Module, criterion: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                       mixup_fn: Optional[Mixup] = None, log_writer=None, args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_supcon', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    
    supcon_criterion = SupConLoss(temperature=0.15).to(device)
    lambda_weight = 0.1 # 你可以在 args 里增加一个控制 SupCon 权重的超参数，这里暂定 0.2

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets, _ = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 记录硬标签用于 SupCon Loss（Mixup 会把 target 变软）
        hard_labels = targets.clone()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            # Mixup 后的 targets shape 是 [B, C]，找回 hard labels
            if len(targets.shape) > 1:
                hard_labels = targets.argmax(dim=1)

        # ---------------- 核心: 多视图生成与联合优化 ----------------
        # 1. 生成物理畸变视图
        samples_phy = physical_augmentation(samples, num_packets=5, mask_prob=0.3, shuffle_prob=0.3)

        # with amp_autocast():
        # 2. 前向传播提取特征
        outputs_orig, z_orig, v_orig = model(samples, return_features=True)
        _, _, v_phy = model(samples_phy, return_features=True)
        
        # 3. 提取统计特征 (绕过 Encoder 节省显存和计算时间)
        z_stat = statistical_augmentation(z_orig, alpha=0.05)
        # 处理 DDP 模型封装
        model_proj = model.module.projector if hasattr(model, 'module') else model.projector
        v_stat = model_proj(z_stat)

        # 4. 计算 Loss
        loss_ce = criterion(outputs_orig, targets)
        
        # 将三种视图在对比空间中拼接并归一化
        features = torch.cat([v_orig, v_phy, v_stat], dim=0)
        features = F.normalize(features, dim=1) # SupCon 计算前必须 L2 归一化！
        
        labels_concat = torch.cat([hard_labels, hard_labels, hard_labels], dim=0)
        loss_supcon = supcon_criterion(features, labels_concat)
        
        # 联合损失
        loss = loss_ce + lambda_weight * loss_supcon
        # ------------------------------------------------------------

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_supcon=loss_supcon.item())
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/total', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss/ce', loss_ce.item(), epoch_1000x)
            log_writer.add_scalar('loss/supcon', loss_supcon.item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_dg_new(model: torch.nn.Module, criterion: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                       mixup_fn: Optional[Mixup] = None, log_writer=None, args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cls', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_pull', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_push', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_orth', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    
    # 初始化损失函数
    supcon_criterion = SupConLoss(temperature=0.15).to(device)
    push_criterion = StyleDivergencePushLoss().to(device)
    orth_criterion = OrthogonalIsolationPenalty().to(device)
    
    # 获取超参数 (建议将其加入 args 中，这里给定默认示例)
    lambda_1 = getattr(args, 'lambda_1', 0.1) # L_pull 权重
    lambda_2 = getattr(args, 'lambda_2', 0.1) # L_push 权重
    lambda_3 = getattr(args, 'lambda_3', 0.1) # L_orth 权重

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets, _ = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 记录硬标签用于 SupCon Loss (Mixup 会软化 target)
        hard_labels = targets.clone()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            if len(targets.shape) > 1:
                hard_labels = targets.argmax(dim=1)

        # ---------------- 核心: 多视图生成与双分支联合优化 ----------------
        # 视图1：原始视图 (X)
        outputs_orig, feat_orig, z_con_orig, z_sty_orig = model(samples, return_features=True)
        
        # 视图2：物理畸变视图 (模拟丢包/乱序)
        samples_phy = physical_augmentation(samples, num_packets=5, mask_prob=0.3, shuffle_prob=0.3)
        _, _, z_con_phy, _ = model(samples_phy, return_features=True)

        # 视图3：统计漂移视图 (在底层 Encoder 提取的特征空间上做统计增强)
        feat_stat = statistical_augmentation(feat_orig, alpha=0.05)
        # 获取封装情况下的 head
        content_head = model.module.content_head if hasattr(model, 'module') else model.content_head
        style_head = model.module.style_head if hasattr(model, 'module') else model.style_head
        
        # 分别通过内容头和风格头
        z_con_stat = content_head(feat_stat)
        z_sty_stat = style_head(feat_stat)

        # ---- 计算四大损失 ----
        # 1. 纯净分类损失 L_cls (Task Classification Loss)
        loss_cls = criterion(outputs_orig, targets)
        
        # 2. 内容不变性拉近损失 L_pull (Supervised Contrastive Learning)
        # 仅针对内容子空间特征 (z_con)
        features_pull = torch.cat([z_con_orig, z_con_phy, z_con_stat], dim=0)
        features_pull = F.normalize(features_pull, dim=1) # SupCon 前必须 L2 归一化
        labels_concat = torch.cat([hard_labels, hard_labels, hard_labels], dim=0)
        loss_pull = supcon_criterion(features_pull, labels_concat)
        
        # 3. 风格发散性推斥损失 L_push (Style Divergence Push Loss)
        # 强制原始视图与统计漂移视图在风格空间相互排斥
        loss_push = push_criterion(z_sty_orig, z_sty_stat)
        
        # 4. 正交隔离惩罚 L_orth (Orthogonal Isolation Penalty)
        # 彻底隔离因果内容与环境风格特征
        loss_orth = orth_criterion(z_con_orig, z_sty_orig)
        
        # 5. 总体优化目标
        loss = loss_cls + lambda_1 * loss_pull + lambda_2 * loss_push + lambda_3 * loss_orth
        # ------------------------------------------------------------

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # 更新记录仪
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_pull=loss_pull.item())
        metric_logger.update(loss_push=loss_push.item())
        metric_logger.update(loss_orth=loss_orth.item())
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/total', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss/cls', loss_cls.item(), epoch_1000x)
            log_writer.add_scalar('loss/pull', loss_pull.item(), epoch_1000x)
            log_writer.add_scalar('loss/push', loss_push.item(), epoch_1000x)
            log_writer.add_scalar('loss/orth', loss_orth.item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def pretrain_one_epoch(model: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler, amp_autocast,
                       log_writer=None,
                       model_without_ddp=None,
                       args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('steps', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    steps_of_one_epoch = len(data_loader)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        steps = steps_of_one_epoch * epoch + data_iter_step
        metric_logger.update(steps=int(steps))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples, label = data

        samples = samples.to(device, non_blocking=True)

        with amp_autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        if args.output_dir and steps % args.save_steps_freq == 0 and epoch > 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='step' + str(steps))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_all = []
    target_all = []

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets, _ = data
        # print('----------------------data---------------')
        # print('targets_shape: ', targets.shape)
        # print('label: ', label)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print('--------------samples-----------------------')
        # print(samples)
        # print('samples_shape: ', samples.shape)
        # print('--------------label-----------------------')
        # print(targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with amp_autocast():
        outputs = model(samples)
        # print('oooo',outputs.shape)
        loss = criterion(outputs, targets)
        
        # print('--------------outputs-----------------------')
        # print(outputs)
        # print('outputs_shape: ', outputs.shape)

        value, pred = outputs.topk(1, 1, True, True)
        # print('pred_shape: ', pred.shape)
        pred = pred.t()
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())

        # print('---------------topk----------------')
        # print('outputs.topk(1, 1, True, True): ', outputs.topk(1, 1, True, True))
        # print('outputs.topk(5, 1, True, True): ', outputs.topk(5, 1, True, True))
        # print('value_shape: ', value.shape)
        # print('pred: ', pred)
        # print('pred_shape: ', pred.shape)
        # print('pred_all: ', pred_all)
        # print('pred_all_shape: ', pred_all.shape)
        # print('target_all: ', target_all)
        # print('target_all_shape: ', target_all.shape)

        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.abs().mean().item())
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, print_label_flag=False, if_stat=False):
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    pred_all = []
    target_all = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target, path = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        output = model(images)
        loss = criterion(output + 1e-8, target)
        # print('-----------------------target--------------------')
        # print(target)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        pred_all.extend(pred[0].cpu())
        target_all.extend(target.cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # print('---------------topk----------------')
        # print('outputs.topk(1, 1, True, True): ', output.topk(1, 1, True, True))
        # print('outputs.topk(5, 1, True, True): ', output.topk(5, 1, True, True))
        # print('target：', target)

        if print_label_flag:

            print('------------start_write_log----------------------')
            log_stats = {'outputs.topk(1, 1, True, True): ': output.topk(1, 1, True, True),
                         'outputs.topk(5, 1, True, True): ': output.topk(5, 1, True, True),
                         'target: ': target,
                         'path：': path}

            # 打开文件写入
            with open(os.path.join("output_finetuning_pretrain_add_cstnet_tls_test_8block_ex_lable_2/cstnet-tls_1.3",
                                   "top_log.txt"), mode="a") as f:
                for key, value in log_stats.items():
                    f.write(f"{key}\n")

                    if isinstance(value, tuple):
                        # 处理 topk 结果，value 是一个元组，包含 (values, indices)
                        values, indices = value

                        # 将 GPU 张量移动到 CPU
                        values = values.cpu().numpy()
                        indices = indices.cpu().numpy()

                        f.write("Values:\n")
                        np.savetxt(f, values, fmt='%.4f')

                        f.write("Indices:\n")
                        np.savetxt(f, indices, fmt='%d')

                    elif isinstance(value, torch.Tensor):
                        # 将 GPU 张量移动到 CPU
                        value = value.cpu().numpy()
                        np.savetxt(f, value, fmt='%d')
                    elif isinstance(value, list):
                        # 处理字符串列表
                        for item in value:
                            f.write(f"{item}\n")

                    f.write("\n")

            print("----------------Log stats saved to log_stats.txt----------------")

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item() / 100, n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item() / 100, n=batch_size)

    macro = precision_recall_fscore_support(target_all, pred_all, average='weighted')
    cm = confusion_matrix(target_all, pred_all)

    # compute acc, precision, recall, f1 for each class
    acc = accuracy_score(target_all, pred_all)
    pre_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(target_all,
                                                                                                    pred_all,
                                                                                                    average=None)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(
        '* Pre {macro_pre:.4f} Rec {macro_rec:.4f} F1 {macro_f1:.4f}'
        .format(macro_pre=macro[0], macro_rec=macro[1],
                macro_f1=macro[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['weighted_pre'] = macro[0]
    test_state['weighted_rec'] = macro[1]
    test_state['weighted_f1'] = macro[2]
    test_state['cm'] = cm
    test_state['acc'] = acc
    test_state['pre_per_class'] = pre_per_class
    test_state['rec_per_class'] = rec_per_class
    test_state['f1_per_class'] = f1_per_class
    test_state['support_per_class'] = support_per_class

    return test_state


import time
import gc


@torch.no_grad()
def evaluate_speed_test(data_loader, model, device, args):
    # switch to evaluation mode
    model.eval()
    model_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    res_list = []
    for i in tqdm(range(4, 11), desc="Batch size"):
        # reset memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        batch_size = 2 ** i
        data_loader_tmp = torch.utils.data.DataLoader(data_loader.dataset, sampler=data_loader.sampler,
                                                      batch_size=batch_size, shuffle=False, num_workers=10,
                                                      pin_memory=True, drop_last=False)
        pred_all = []
        start_time = time.time()
        for batch in data_loader_tmp:
            if time.time() - start_time > 30:
                break
            images = batch[0]
            # target = batch[-1]
            images = images.to(device, non_blocking=True)
            # target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                # loss = criterion(output, target)

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            pred_all.extend(pred[0].cpu())
            # target_all.extend(target.cpu())
        end_time = time.time()
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.empty_cache()
        gc.collect()
        res_list.append({
            "batch size": batch_size,
            "time (s)": end_time - start_time,
            "total smaples": len(pred_all),
            "speed (sample per second)": len(pred_all) / (end_time - start_time),
            "max memory consumption (MB)": max_mem,
            "model memory consumption (MB)": model_mem
        })

    with open(os.path.join(args.output_dir, "speed_test.json"), "w") as f:
        json.dump(res_list, f, indent=2)

    return None
