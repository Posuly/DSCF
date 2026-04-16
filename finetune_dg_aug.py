import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchvision import datasets, transforms
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import count_parameters

import models_net_mamba_mambavision
import models_net_mamba_dg as models_net_mamba
from contextlib import suppress

###############aug#################
from engine import train_one_epoch_dg, evaluate
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for traffic classification (Domain Generalization)', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_steps_freq', default=5000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='net_mamba_classifier', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=40, type=int, help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR', help='base learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='AutoAugment policy')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup/cutmix')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix')
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix')

    # Paths
    parser.add_argument('--finetune', default='././output/pretrain/checkpoint.pth', help='finetune from checkpoint')
    
    # NOTE: 修改数据路径参数
    parser.add_argument('--data_path', default='./dataset/Final_Source_Android', type=str, help='Source Domain dataset path')
    parser.add_argument('--target_data_path', default='./dataset/Final_Target_iOS', type=str, help='Target Domain dataset path (for testing only)')
    
    parser.add_argument('--nb_classes', default=14, type=int, help='number of classes')
    parser.add_argument('--output_dir', default='./output/finetune_dg', help='path where to save')
    parser.add_argument('--log_dir', default='./output/finetune_dg', help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # AMP
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    return parser

def normalize_array(tensor, mean, std, dtype=torch.float32):
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero.")
    if mean.ndim == 1: mean = mean.view(-1, 1)
    if std.ndim == 1: std = std.view(-1, 1)
    return tensor.sub_(mean).div_(std)

def min_max_normalize(tensor):
    return (tensor) / (255)

class NPYPipelineDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._make_dataset()
        self.label_to_idx = self._get_label_to_idx()

    def _make_dataset(self):
        samples = []
        if not os.path.exists(self.root_dir):
            print(f"Warning: Directory {self.root_dir} does not exist.")
            return []
            
        for root, _, fnames in sorted(os.walk(self.root_dir)):
            for fname in sorted(fnames):
                if fname.endswith('.npy'):
                    path = os.path.join(root, fname)
                    label = os.path.basename(os.path.dirname(path))
                    samples.append((path, label))
        
        if not samples:
            print(f"Warning: No .npy files found in {self.root_dir}")
        return samples

    def _get_label_to_idx(self):
        if not self.samples:
            return {}
        labels = sorted(set(label for _, label in self.samples))
        # print({label: idx for idx, label in enumerate(labels)})
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        data = min_max_normalize(data)

        if self.transform is not None:
            mean = [0.5]
            std = [0.5]
            data = normalize_array(data, mean, std)

        label_idx = self.label_to_idx[label]
        return data, label_idx, path

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    class ToTensor(object):
        def __call__(self, x):
            return torch.tensor(x, dtype=torch.float32)

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        def __call__(self, x):
            return (x - self.mean) / self.std

    transform_train = transforms.Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
    ])

    # --- NOTE: 构建 Source Domain 数据集 (用于训练和验证模型选择) ---
    print(f"Loading Source Domain Data from: {args.data_path}")
    dataset_train_src = NPYPipelineDataset(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val_src   = NPYPipelineDataset(os.path.join(args.data_path, 'valid'), transform=transform_train)
    dataset_test_src  = NPYPipelineDataset(os.path.join(args.data_path, 'test'), transform=transform_train)
    
    print(f"src.label_to_idx: {dataset_train_src.label_to_idx}")

    # --- NOTE: 构建 Target Domain 数据集 (仅用于测试泛化能力) ---
    print(f"Loading Target Domain Data from: {args.target_data_path}")
    # Target 域通常没有 train/valid 划分，或者我们直接用它的 test 集来评估
    # 如果 Target 文件夹结构也是 train/valid/test，我们可以把它们合并，或者只用 test
    # 这里假设我们只关心 Target 的测试集表现
    dataset_test_tgt = NPYPipelineDataset(os.path.join(args.target_data_path, 'test'), transform=transform_train)
    print(f"tgt.label_to_idx: {dataset_test_tgt.label_to_idx}")

    
    # 如果 Target 的 test 没数据，尝试读取 train (视你的数据集整理情况而定)
    if len(dataset_test_tgt) == 0:
        print("Target test set is empty, trying to load 'train' as test set...")
        dataset_test_tgt = NPYPipelineDataset(os.path.join(args.target_data_path, 'train'), transform=transform_train)

    print(f"Source Train Samples: {len(dataset_train_src)}")
    print(f"Source Valid Samples: {len(dataset_val_src)}")
    print(f"Target Test Samples:  {len(dataset_test_tgt)}")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Samplers
    sampler_train_src = torch.utils.data.DistributedSampler(
        dataset_train_src, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args.dist_eval:
        if len(dataset_val_src) % num_tasks != 0:
            print('Warning: Dist eval dataset not divisible by process number.')
        sampler_val_src = torch.utils.data.DistributedSampler(dataset_val_src, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_test_tgt = torch.utils.data.DistributedSampler(dataset_test_tgt, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val_src = torch.utils.data.SequentialSampler(dataset_val_src)
        sampler_test_tgt = torch.utils.data.SequentialSampler(dataset_test_tgt)

    # DataLoaders
    # 1. Source Train
    data_loader_train_src = torch.utils.data.DataLoader(
        dataset_train_src, sampler=sampler_train_src,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )

    # 2. Source Valid (用于挑选最佳模型)
    data_loader_val_src = torch.utils.data.DataLoader(
        dataset_val_src, sampler=sampler_val_src,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    # 3. Target Test (用于评估 OOD 泛化性能)
    data_loader_test_tgt = torch.utils.data.DataLoader(
        dataset_test_tgt, sampler=sampler_test_tgt,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Model
    model = models_net_mamba.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    # Load Pretrained
    if os.path.exists(args.finetune) and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Optimizer
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = "none"
    amp_autocast = suppress
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        print("Start evaluating on Target Domain...")
        test_stats = evaluate(data_loader_test_tgt, model, device)
        print(f"Target Domain Accuracy: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    max_accuracy_src = 0.0
    best_acc_src = 0
    best_epoch = 0
    
    print("Train dataset size:", len(dataset_train_src))
    print("Batch size:", args.batch_size)
    print("Train loader length:", len(data_loader_train_src))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train_src.sampler.set_epoch(epoch)
        
        # 1. Train on Source
        train_stats = train_one_epoch_dg(
            model, criterion, data_loader_train_src,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        # 2. Evaluate on Source Valid (In-Domain) - Used for model selection
        print(f'--- Epoch {epoch}: Validating on Source Domain ---')
        val_stats_src = evaluate(data_loader_val_src, model, device, print_label_flag=False)
        print(f"Source Valid Acc: {val_stats_src['acc1']:.4f}")

        # 3. Evaluate on Target Test (Out-of-Domain) - Used for DG monitoring
        print(f'--- Epoch {epoch}: Testing on Target Domain (DG) ---')
        test_stats_tgt = evaluate(data_loader_test_tgt, model, device, print_label_flag=False)
        print(f"Target Test Acc:  {test_stats_tgt['acc1']:.4f}")

        # Save Best Model based on SOURCE Metric (DG standard)
        # 域泛化通常假设我们不知道 Target，所以只能根据 Source Valid 来选模型
        if val_stats_src["acc"] > best_acc_src:
            best_acc_src = val_stats_src["acc"]
            best_epoch = epoch
            misc.save_model(args=args, model=model, model_without_ddp=model,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            name="best") # save best_checkpoint

        # 也可以每隔一定步数保存一个 checkpoint
        if epoch % 20 == 0 or epoch + 1 == args.epochs:
             misc.save_model(args=args, model=model, model_without_ddp=model,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            name=f"epoch_{epoch}")

        # Logging
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'src_val_{k}': v for k, v in val_stats_src.items() if not isinstance(v, (np.ndarray, list))},
            **{f'tgt_test_{k}': v for k, v in test_stats_tgt.items() if not isinstance(v, (np.ndarray, list))},
            'n_parameters': n_parameters
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.add_scalar('perf/src_val_acc', val_stats_src['acc1'], epoch)
                log_writer.add_scalar('perf/tgt_test_acc', test_stats_tgt['acc1'], epoch)
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Finally, load best model and report final results
    print("Loading best model (selected by Source Valid) for final evaluation...")
    best_checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"), map_location='cpu')
    model.load_state_dict(best_checkpoint['model'])

    # Final Eval on Source Test
    final_src_res = evaluate(DataLoader(dataset_test_src, batch_size=args.batch_size), model, device)
    # Final Eval on Target Test
    final_tgt_res = evaluate(data_loader_test_tgt, model, device)

    print(f"Final Source Test Acc: {final_src_res['acc1']:.4f}")
    print(f"Final Target Test Acc (DG Performance): {final_tgt_res['acc1']:.4f}")

    with open(os.path.join(args.output_dir, "final_stats.json"), mode="w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "source_test_acc": final_src_res["acc1"],
            "target_test_acc": final_tgt_res["acc1"],
            "source_test_f1": final_src_res["weighted_f1"],
            "target_test_f1": final_tgt_res["weighted_f1"],
        }, f, indent=2)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)