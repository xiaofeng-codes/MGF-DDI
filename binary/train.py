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

# 导入你的数据加载器和模型
from data_preprocess import load_data
from model import MGF_DDI

# ==========================================
# 1. 配置与参数 (Configuration)
# ==========================================
parser = argparse.ArgumentParser(description="Training Script for MGF-DDI (Binary Classification)")

# 数据与路径
parser.add_argument('--dataset', type=str, default='zhang', choices=['zhang', 'miner', 'deep'], help='Dataset name')
parser.add_argument('--dataset_dir', type=str, default='dataset/', help='Directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='Fold index for cross-validation')
parser.add_argument('--save_dir', type=str, default='results/', help='Directory to save results')

# 模型超参数
parser.add_argument('--n_atom_feats', type=int, default=36, help='Input atom feature dimension')
parser.add_argument('--micro_layers', type=int, default=3, help='Number of GCN layers')
parser.add_argument('--gt_layers', type=int, default=3, help='Number of Graph Transformer layers')
parser.add_argument('--mem_num', type=int, default=6, help='Number of memory slots')

# 训练超参数
parser.add_argument('--n_epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--lambda_aux', type=float, default=0.1, help='Weight for auxiliary losses')

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

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)         # 大于0.5为1，否则置0
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    # p, r, t = metrics.precision_recall_curve(target, probas_pred)
    # int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, recall, precision,ap


def compute_metrics(logits, targets):
    """
    计算二分类的6个指标: ACC, AUC, F1, Recall, Precision, AP
    Args:
        logits: 模型原始输出 (未经过 Sigmoid), shape (N, 1) 或 (N,)
        targets: 真实标签, shape (N, 1) 或 (N,)
    """
    # 确保转换为一维数组
    logits = np.array(logits).flatten()
    targets = np.array(targets).flatten()

    # 1. 计算概率 (Sigmoid)
    probas = 1 / (1 + np.exp(-logits))

    # 2. 预测类别 (阈值 0.5)
    preds = (probas > 0.5).astype(int)

    # 3. 计算指标
    # ACC
    acc = metrics.accuracy_score(targets, preds)

    # Precision (Pr)
    precision = metrics.precision_score(targets, preds, average='binary', zero_division=0)

    # Recall (Rec)
    recall = metrics.recall_score(targets, preds, average='binary', zero_division=0)

    # F1
    f1 = metrics.f1_score(targets, preds, average='binary', zero_division=0)

    # AUC (ROC-AUC)
    try:
        auc = metrics.roc_auc_score(targets, probas)
    except ValueError:
        auc = 0.0

    # AP (Average Precision score, 对应 PR 曲线下的面积)
    try:
        ap = metrics.average_precision_score(targets, probas)
    except ValueError:
        ap = 0.0

    return acc, auc, f1, recall, precision, ap


# ==========================================
# 3. 训练器 (Trainer)
# ==========================================
class Trainer:
    def __init__(self, model, optimizer, scheduler, device, criterion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion  # BCEWithLogitsLoss

    def run_epoch(self, data_loader, epoch, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        losses = AverageMeter()
        all_logits = []
        all_targets = []

        # Tqdm 进度条
        pbar = tqdm(data_loader, desc=f"{mode.capitalize()} Epoch {epoch}", ncols=120)

        for batch in pbar:
            # 1. 解包数据
            h_macro, t_macro, h_motif, t_motif, rels = [item.to(self.device) for item in batch]

            # 标签处理: 转 float 并调整维度 (Batch) -> (Batch, 1)
            labels = rels.float().unsqueeze(1)

            # 2. 前向传播
            if mode == 'train':
                self.optimizer.zero_grad()
                logits, aux_loss = self.model(h_macro, t_macro, h_motif, t_motif)

                # 计算损失
                pred_loss = self.criterion(logits, labels)
                total_loss = pred_loss + args.lambda_aux * aux_loss

                # 反向传播
                total_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits, aux_loss = self.model(h_macro, t_macro, h_motif, t_motif)
                    pred_loss = self.criterion(logits, labels)
                    total_loss = pred_loss + args.lambda_aux * aux_loss

            # 3. 记录 Loss
            losses.update(total_loss.item(), rels.size(0))

            # 4. 收集结果 (detach to cpu)
            all_logits.append(torch.sigmoid(logits.detach()).cpu())
            all_targets.append(labels.detach().cpu())

            pbar.set_postfix({'loss': f"{losses.avg:.4f}"})

        # 5. 计算整轮指标
        train_probas_pred = np.concatenate(all_logits)
        train_ground_truth = np.concatenate(all_targets)
        acc, auc, f1, rec, pre, ap = do_compute_metrics(train_probas_pred, train_ground_truth)

        return losses.avg, acc, auc, f1, rec, pre, ap


# ==========================================
# 4. 主程序 (Main)
# ==========================================
def main():
    # --- 环境设置 ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 数据加载 ---
    train_loader, val_loader, test_loader = load_data(args, args.batch_size, args.fold)

    # --- 模型初始化 ---
    # 确保 model.py 中的 MGF_DDI 已调整为二分类 (输出维度 1)
    num_node_feats_dict = {"zhang": 30, "miner": 32, "deep": 41}
    n_atom_feats = num_node_feats_dict[args.dataset]

    # --- 模型初始化 ---
    model = MGF_DDI(
        d_atom=n_atom_feats,
        micro_layers=args.micro_layers,
        gt_layers=args.gt_layers,
        mem_num=args.mem_num,
        dropout=args.dropout
    ).to(device)

    # --- 优化器与损失 ---
    # 二分类使用 BCEWithLogitsLoss (内置 Sigmoid)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    trainer = Trainer(model, optimizer, scheduler, device, criterion)

    # --- 记录文件 ---
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"{args.dataset}_fold{args.fold}_{timestamp}.csv")
    best_model_path = os.path.join(args.save_dir, f"{args.dataset}_fold{args.fold}_best.pth")

    # CSV 表头包含所有6个指标
    results_cols = [
        'Epoch',
        'Train Loss', 'Train ACC', 'Train AUC',
        'Val Loss', 'Val ACC', 'Val AUC', 'Val F1', 'Val Rec', 'Val Pre', 'Val AP'
    ]
    results_df = pd.DataFrame(columns=results_cols)
    results_df.to_csv(csv_path, index=False)

    # --- 训练循环 ---
    best_metric = 0.0  # 通常基于 AUC 或 AP 选择最佳模型

    logger.info("Start Training...")
    for epoch in range(1, args.n_epochs + 1):
        start_time = time.time()

        # Train (只记录主要指标以保持日志简洁，但你可以按需全记录)
        t_loss, t_acc, t_auc, t_f1, t_rec, t_pre, t_ap = trainer.run_epoch(train_loader, epoch, mode='train')

        # Val
        v_loss, v_acc, v_auc, v_f1, v_rec, v_pre, v_ap = trainer.run_epoch(val_loader, epoch, mode='val')

        scheduler.step()
        duration = time.time() - start_time

        # 打印日志
        logger.info(f"Epoch {epoch:03d} | Time: {duration:.2f}s")
        logger.info(f"Train | Loss:{t_loss:.4f} | ACC:{t_acc:.4f} | AUC:{t_auc:.4f} | F1:{t_f1:.4f} | Rec:{t_rec:.4f} | Pre:{t_pre:.4f} | AP:{t_ap:.4f}")
        logger.info(
            f"Val   | Loss:{v_loss:.4f} | ACC:{v_acc:.4f} | AUC:{v_auc:.4f} | F1:{v_f1:.4f} | Rec:{v_rec:.4f} | Pre:{v_pre:.4f} | AP:{v_ap:.4f}")

        # 保存最佳模型 (这里以 AUC 为准，也可以改为 v_ap 或 v_acc)
        if v_acc > best_metric:
            best_metric = v_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"--> Best Model Saved (ACC: {best_metric:.4f})")

        # 写入CSV
        row_stats = pd.DataFrame([[
            epoch,
            t_loss, t_acc, t_auc,
            v_loss, v_acc, v_auc, v_f1, v_rec, v_pre, v_ap
        ]], columns=results_cols)
        row_stats.to_csv(csv_path, mode='a', header=False, index=False)

    # --- 测试 ---
    logger.info("Training Finished. Running Testing...")
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_auc, test_f1, test_rec, test_pre, test_ap = trainer.run_epoch(test_loader, 0, mode='test')

    logger.info("================ Test Results ================")
    logger.info(f"ACC: {test_acc:.4f} | AUC: {test_auc:.4f} | AP: {test_ap:.4f}")
    logger.info(f"F1 : {test_f1:.4f} | Rec: {test_rec:.4f} | Pre: {test_pre:.4f}")

    # 将测试结果追加到 CSV
    with open(csv_path, 'a') as f:
        f.write(f"\nTEST RESULT\n")
        f.write(f"Loss,ACC,AUC,F1,Rec,Pre,AP\n")
        f.write(f"{test_loss:.4f},{test_acc:.4f},{test_auc:.4f},{test_f1:.4f},{test_rec:.4f},{test_pre:.4f},{test_ap:.4f}\n")



if __name__ == '__main__':
    main()