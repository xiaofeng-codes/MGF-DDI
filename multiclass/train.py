import os
import time
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn import metrics
from tqdm import tqdm
from datetime import datetime

# 假设 data_pre 已经包含了我们之前的优化
from data_preprocess import load_data

# 假设 model 文件中定义了 MGF_DDI 模型
# from model import MGF_DDI

# ==========================================
# 1. 配置与参数 (Configuration)
# ==========================================
parser = argparse.ArgumentParser(description="Training Script for MGF-DDI")

# 数据与路径
parser.add_argument('--dataset', type=str, default='deng', choices=['deng', 'drugbank', 'deep'], help='Dataset name')
parser.add_argument('--fold', type=int, default=0, help='Fold index for cross-validation')
parser.add_argument('--save_dir', type=str, default='results/', help='Directory to save results')

# 模型超参数
parser.add_argument('--n_atom_feats', type=int, default=36, help='Input atom feature dimension')
parser.add_argument('--gt_layers', type=int, default=3, help='Number of GNN layers')
parser.add_argument('--micro_layers', type=int, default=3, help='Number of Graph Transformer layers')
parser.add_argument('--mem_num', type=int, default=6, help='Number of memory slots')
parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--rel_total', type=int, default=86, help='Number of interaction types (classes)')

# 训练超参数
parser.add_argument('--n_epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--lambda_aux', type=float, default=0.1,
                    help='Weight for auxiliary losses (Distillation/Contrastive)')

# 设备
parser.add_argument('--gpu', type=int, default=0, help='GPU index')

args = parser.parse_args()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 2. 工具类 (Utils)
# ==========================================
class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(probas, targets):
    """计算多分类指标"""
    preds = np.argmax(probas, axis=1)

    acc = metrics.accuracy_score(targets, preds)
    f1 = metrics.f1_score(targets, preds, average='macro', zero_division=0)
    precision = metrics.precision_score(targets, preds, average='macro', zero_division=0)
    recall = metrics.recall_score(targets, preds, average='macro', zero_division=0)

    return acc, f1, precision, recall


# ==========================================
# 3. 训练器 (Trainer)
# ==========================================
class Trainer:
    def __init__(self, model, optimizer, scheduler, device, criterion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion  # 主任务损失 (CE)

    def run_epoch(self, data_loader, epoch, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        losses = AverageMeter()
        all_preds = []
        all_targets = []

        # Tqdm 进度条
        pbar = tqdm(data_loader, desc=f"{mode.capitalize()} Epoch {epoch}", ncols=120)

        for batch in pbar:
            # 1. 解包数据 (对应 data_pre.py 的 collate_fn)
            # h_macro/t_macro: 宏观图 (Batch对象)
            # h_motif/t_motif: 模体图 (Batch对象)
            # rels: 标签
            h_macro, t_macro, h_motif, t_motif, rels = [item.to(self.device) for item in batch]

            # 2. 前向传播
            if mode == 'train':
                self.optimizer.zero_grad()
                # 假设模型返回: (分类Logits, 辅助Loss)
                # 如果模型还没实现辅助Loss，aux_loss 返回 0
                logits, aux_loss = self.model(h_macro, t_macro, h_motif, t_motif)

                # 3. 计算总损失
                pred_loss = self.criterion(logits, rels)
                total_loss = pred_loss + args.lambda_aux * aux_loss

                # 4. 反向传播
                total_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits, aux_loss = self.model(h_macro, t_macro, h_motif, t_motif)
                    pred_loss = self.criterion(logits, rels)
                    total_loss = pred_loss + args.lambda_aux * aux_loss

            # 5. 记录
            losses.update(total_loss.item(), rels.size(0))
            all_preds.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_targets.append(rels.detach().cpu().numpy())

            pbar.set_postfix({'loss': f"{losses.avg:.4f}"})

        # 6. 计算Epoch指标
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc, f1, prec, rec = compute_metrics(all_preds, all_targets)

        return losses.avg, acc, f1, prec, rec


# ==========================================
# 4. 主程序 (Main)
# ==========================================
def main():
    # --- 环境设置 ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Hyperparameters: {args}")

    # --- 数据加载 ---
    # 这里的 load_data 来自优化后的 data_pre.py
    train_loader, val_loader, test_loader = load_data(args, args.batch_size, args.fold)

    # --- 模型初始化 ---
    from model import MGF_DDI
    model = MGF_DDI(
        d_atom=args.n_atom_feats,
        micro_layers=args.micro_layers,
        gt_layers=args.gt_layers,
        mem_num=args.mem_num,
        dropout=args.dropout
    ).to(device)

    # --- 优化器与损失 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    trainer = Trainer(model, optimizer, scheduler, device, criterion)

    # --- 记录文件 ---
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"{args.dataset}_fold{args.fold}_{timestamp}.csv")
    best_model_path = os.path.join(args.save_dir, f"{args.dataset}_fold{args.fold}_best.pkl")

    results_cols = ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Val F1', 'Val Recall']
    results_df = pd.DataFrame(columns=results_cols)
    results_df.to_csv(csv_path, index=False)

    # --- 训练循环 ---
    best_acc = 0.0

    logger.info("Start Training...")
    for epoch in range(1, args.n_epochs + 1):
        start_time = time.time()

        # Train
        t_loss, t_acc, t_f1, t_prec, t_rec = trainer.run_epoch(train_loader, epoch, mode='train')

        # Val
        v_loss, v_acc, v_f1, v_prec, v_rec = trainer.run_epoch(val_loader, epoch, mode='val')

        scheduler.step()
        duration = time.time() - start_time

        # 打印日志
        logger.info(f"Epoch {epoch:03d} | Time: {duration:.2f}s")
        logger.info(f"Train | Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | F1: {t_f1:.4f} | Precision: {t_prec:.4f} | Recall: {t_rec:.4f}")
        logger.info(f"Val   | Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f} | Precision: {v_prec:.4f} | Recall: {v_rec:.4f}")

        # 保存最佳模型
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), best_model_path)  # 推荐只保存 state_dict
            logger.info(f"--> Best Model Saved (Acc: {best_acc:.4f})")

        # 写入CSV
        row_stats = pd.DataFrame([[epoch, t_loss, t_acc, v_loss, v_acc, v_f1, v_rec]], columns=results_cols)
        row_stats.to_csv(csv_path, mode='a', header=False, index=False)

    # --- 测试 ---
    logger.info("Training Finished. Running Testing...")
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1, test_prec, test_rec = trainer.run_epoch(test_loader, 0, mode='test')

    logger.info("================ Test Results ================")
    logger.info(f"Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f}")

    # 将测试结果追加到 CSV
    with open(csv_path, 'a') as f:
        f.write(f"\nTest Results\n")
        f.write(f"Loss,Acc,F1,Precision,Recall\n")
        f.write(f"{test_loss:.4f},{test_acc:.4f},{test_f1:.4f},{test_prec:.4f},{test_rec:.4f}\n")


if __name__ == '__main__':
    main()