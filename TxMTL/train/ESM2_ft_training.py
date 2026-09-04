import argparse
import torch
import os
from torch import nn
from transformers import EsmModel, EsmTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, 
                            f1_score, roc_auc_score, matthews_corrcoef, 
                            average_precision_score, confusion_matrix)

from MTL_model import MultiTask_ESM_Classifier  # 修改导入
import random

# 配置参数
MAX_LEN = 2000
NUM_TASKS = 7  # 7种分泌系统类型
TASK_MAPPING = {
    'T1SE': 0, 'T2SE': 1, 'T3SE': 2, 'T4SE': 3, 
    'T5SE': 4, 'T6SE': 5, 'T7SE': 6
}
REVERSE_TASK_MAPPING = {v: k for k, v in TASK_MAPPING.items()}

# 修改：使用ESM-2的tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=3, verbose=True, delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
               self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weights=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weights,reduction=reduction)
        self.gamma = gamma
        self.weights = weights.float() #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def alpha_calc(train_set_csv_path):
    """计算每个任务的正负样本比例，用于Focal Loss"""
    training_data = pd.read_csv(train_set_csv_path)
    
    # 统计每个任务的样本数量
    task_counts = training_data['task'].value_counts()
    
    alpha_list = []
    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        
        if task_name in task_counts:
            # 该任务的样本数为正样本数
            pos_count = task_counts[task_name]
            # 总样本数减去正样本数为负样本数
            total_samples = len(training_data)
            neg_count = total_samples - pos_count
            
            # 计算alpha = 负样本比例
            alpha = neg_count / total_samples if total_samples > 0 else 0.5
        else:
            alpha = 0.5  # 默认值
        
        alpha_list.append(alpha)
    
    return alpha_list

class MultiTaskDataset(Dataset):
    """多任务学习数据集"""
    def __init__(self, sequences, task_labels, tokenizer, max_len):
        """
        参数:
            sequences: 蛋白质序列列表 (带空格)
            task_labels: 任务标签列表 (如 'T1SE', 'T2SE'等)
            tokenizer: ESM tokenizer
            max_len: 最大序列长度
        """
        self.sequences = sequences
        self.task_labels = task_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        task_name = str(self.task_labels[idx])
        
        # 获取真实任务ID
        true_task_id = TASK_MAPPING.get(task_name, -1)
        
        # 为所有任务创建标签向量
        # 该样本在真实任务中是正样本(1)，在其他任务中是负样本(0)
        all_task_labels = torch.zeros(NUM_TASKS, dtype=torch.long)
        if true_task_id != -1:
            all_task_labels[true_task_id] = 1
        
        # Tokenize序列 - ESM tokenizer用法与BERT类似

        encoding = tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'sequence': sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'true_task_id': true_task_id,  # 该样本的真实任务ID
            'all_task_labels': all_task_labels,  # 所有任务的标签向量
        }

def create_data_loader(csv_path, batch_size, shuffle=True):
    """创建数据加载器"""
    df = pd.read_csv(csv_path)
    
    # 检查必要列
    required_cols = ['SEQUENCE_space', 'task']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要列: {col}")
    
    # 创建数据集
    dataset = MultiTaskDataset(
        sequences=df['SEQUENCE_space'].tolist(),
        task_labels=df['task'].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader

def freeze_esm_layers(model, num_frozen_layers):
    """冻结ESM的前面几层"""
    if num_frozen_layers > 0:
        for layer in model.esm.encoder.layer[:num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"冻结了 {num_frozen_layers} 层ESM参数")

def calculate_metrics(y_true, y_pred, y_probs):
    """计算评估指标"""
    metrics = {}
    
    if len(np.unique(y_true)) > 1:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            # 基础指标
            metrics['ACC'] = accuracy_score(y_true, y_pred)
            metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['Recall'] = tp / (tp + fn)
            metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
            
            # 需要概率的指标
            metrics['AUC_ROC'] = roc_auc_score(y_true, y_probs)
            metrics['AUC_PRC'] = average_precision_score(y_true, y_probs)
            
            # 特异性
            metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        except Exception as e:
            print(f"计算指标时出错: {e}")
            metrics = {k: 0.0 for k in ['ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUC_ROC', 'AUC_PRC', 'Specificity']}
    else:
        metrics = {k: 0.0 for k in ['ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUC_ROC', 'AUC_PRC', 'Specificity']}
    
    return metrics

def train(args):
    """训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载训练数据...")
    train_loader = create_data_loader(args.train_dir, args.batch_size, shuffle=True)
    print("加载验证数据...")
    val_loader = create_data_loader(args.val_dir, args.batch_size, shuffle=False)
    
    # 初始化模型
    print("初始化模型...")
    model = MultiTask_ESM_Classifier(num_tasks=NUM_TASKS, num_classes_per_task=2)  # 修改模型
    
    # 冻结指定层
    freeze_esm_layers(model, args.frozen_layers)  # 修改函数名
    
    model = model.to(device)
    
    # 优化器 - 针对ESM可能需要不同的学习率
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10%的warmup
        num_training_steps=total_steps
    )
    
    # 为每个任务创建损失函数
    alpha_list = alpha_calc(args.train_dir)
    print(f"各任务alpha值: {alpha_list}")
    
    loss_functions = []
    for task_id in range(NUM_TASKS):
        alpha = alpha_list[task_id]
        class_weights = torch.tensor(
            [1 - alpha, alpha],
            dtype=torch.float32,
            device=device
        )
        loss_fn = FocalLoss(weights=class_weights, gamma=2.0)
        loss_functions.append(loss_fn)
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=args.checkpoint_path
    )
    
    # 存储最佳指标
    best_metrics = {
        'overall_F1': 0.0,
        'overall_AUC': 0.0,
        'task_best_F1': [0.0] * NUM_TASKS,
        'task_best_AUC': [0.0] * NUM_TASKS,
    }
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)
        
        # 训练阶段
        model.train()
        train_losses = []
        train_task_predictions = {task_id: {'true': [], 'pred': [], 'probs': []} 
                                 for task_id in range(NUM_TASKS)}
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            all_task_labels = batch['all_task_labels'].to(device)  # shape: [batch_size, NUM_TASKS]
            
            # 前向传播
            task_outputs = model(input_ids, attention_mask=attention_mask)
            
            # 计算每个任务的损失
            total_loss = 0
            for task_id in range(NUM_TASKS):
                # 该任务的输出
                task_output = task_outputs[task_id]  # shape: [batch_size, 2]
                
                # 该任务的标签（正样本为1，负样本为0）
                task_labels = all_task_labels[:, task_id].long()  # shape: [batch_size]
                
                # 计算损失
                task_loss = loss_functions[task_id](task_output, task_labels)
                total_loss += task_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # 更新学习率
            
            train_losses.append(total_loss.item())
            
            # 收集训练预测
            with torch.no_grad():
                for task_id in range(NUM_TASKS):
                    task_output = task_outputs[task_id]
                    probs = torch.softmax(task_output, dim=1)[:, 1].cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    labels = all_task_labels[:, task_id].cpu().numpy()
                    
                    train_task_predictions[task_id]['true'].extend(labels)
                    train_task_predictions[task_id]['pred'].extend(preds)
                    train_task_predictions[task_id]['probs'].extend(probs)
            
            if (batch_idx + 1) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_train_loss = np.mean(train_losses)
        print(f"训练平均损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_task_predictions = {task_id: {'true': [], 'pred': [], 'probs': []} 
                               for task_id in range(NUM_TASKS)}
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                all_task_labels = batch['all_task_labels'].to(device)
                
                task_outputs = model(input_ids, attention_mask=attention_mask)
                
                total_val_loss = 0
                for task_id in range(NUM_TASKS):
                    task_output = task_outputs[task_id]
                    probs = torch.softmax(task_output, dim=1)[:, 1].cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    labels = all_task_labels[:, task_id]
                    
                    task_val_loss = loss_functions[task_id](task_output, labels)
                    total_val_loss += task_val_loss

                    val_task_predictions[task_id]['true'].extend(labels.cpu().numpy())
                    val_task_predictions[task_id]['pred'].extend(preds)
                    val_task_predictions[task_id]['probs'].extend(probs)
        
                val_losses.append(total_val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f"验证平均损失: {avg_val_loss:.4f}")
        
        # 计算验证集指标
        val_metrics_per_task = []
        overall_val_metrics = {'ACC': [],'Sn':[],'Sp':[],'PR':[],'F1': [],'AUC_ROC': [],'MCC': [],'AUC_PRC':[]}
        
        print("\n验证集指标:")
        print("-" * 60)
        
        for task_id in range(NUM_TASKS):
            task_name = REVERSE_TASK_MAPPING[task_id]
            y_true = np.array(val_task_predictions[task_id]['true'])
            y_pred = np.array(val_task_predictions[task_id]['pred'])
            y_probs = np.array(val_task_predictions[task_id]['probs'])
            
            # 计算该任务的指标
            task_metrics = calculate_metrics(y_true, y_pred, y_probs)
            val_metrics_per_task.append(task_metrics)
            
            # 收集整体指标
            overall_val_metrics['ACC'].append(task_metrics['ACC'])
            overall_val_metrics['Sn'].append(task_metrics['Recall'])
            overall_val_metrics['Sp'].append(task_metrics['Specificity'])
            overall_val_metrics['PR'].append(task_metrics['Precision'])
            overall_val_metrics['F1'].append(task_metrics['F1'])
            overall_val_metrics['AUC_ROC'].append(task_metrics['AUC_ROC'])
            overall_val_metrics['MCC'].append(task_metrics['MCC'])
            overall_val_metrics['AUC_PRC'].append(task_metrics['AUC_PRC'])
            
            # 更新任务最佳指标
            if task_metrics['F1'] > best_metrics['task_best_F1'][task_id]:
                best_metrics['task_best_F1'][task_id] = task_metrics['F1']
                best_metrics['task_best_AUC'][task_id] = task_metrics['AUC_ROC']
            
            print(f"{task_name:5s}: ACC={task_metrics['ACC']:.4f}, Sn={task_metrics['Recall']:.4f}, "
                  f"Sp={task_metrics['Specificity']:.4f}, PR={task_metrics['Precision']:.4f}, "
                  f"F1={task_metrics['F1']:.4f}, AUROC={task_metrics['AUC_ROC']:.4f}, "
                  f"MCC={task_metrics['MCC']:.4f}, AUPRC={task_metrics['AUC_PRC']:.4f}")
        
        # 计算整体平均指标
        overall_avg_ACC = np.mean(overall_val_metrics['ACC'])
        overall_avg_Sn = np.mean(overall_val_metrics['Sn'])
        overall_avg_Sp = np.mean(overall_val_metrics['Sp'])
        overall_avg_PR = np.mean(overall_val_metrics['PR'])
        overall_avg_F1 = np.mean(overall_val_metrics['F1'])
        overall_avg_ROC = np.mean(overall_val_metrics['AUC_ROC'])
        overall_avg_MCC = np.mean(overall_val_metrics['MCC'])
        overall_avg_PRC = np.mean(overall_val_metrics['AUC_PRC'])
        
        print("-" * 60)
        print(f"整体平均: ACC={overall_avg_ACC:.4f}, Sn={overall_avg_Sn:.4f}, Sp={overall_avg_Sp:.4f}, "
              f"PR={overall_avg_PR:.4f}, F1={overall_avg_F1:.4f}, AUROC={overall_avg_ROC:.4f}, "
              f"MCC={overall_avg_MCC:.4f}, AUPRC={overall_avg_PRC:.4f}")
        
        # 更新整体最佳指标
        if overall_avg_F1 > best_metrics['overall_F1']:
            best_metrics['overall_F1'] = overall_avg_F1
            best_metrics['overall_AUC'] = overall_avg_ROC
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"★ 新的最佳F1分数: {overall_avg_F1:.4f}")
        
        # 保存结果
        result_dict = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,  # 添加验证损失
            'overall_ACC': overall_avg_ACC,
            'overall_Sn': overall_avg_Sn,
            'overall_Sp': overall_avg_Sp,
            'overall_PR': overall_avg_PR,
            'overall_F1': overall_avg_F1,
            'overall_AUROC': overall_avg_ROC,
            'overall_MCC': overall_avg_MCC,
            'overall_AUPRC': overall_avg_PRC,
        }
        
        # 添加每个任务的指标
        for task_id in range(NUM_TASKS):
            task_name = REVERSE_TASK_MAPPING[task_id]
            result_dict[f'{task_name}_ACC'] = val_metrics_per_task[task_id]['ACC']
            result_dict[f'{task_name}_Sn'] = val_metrics_per_task[task_id]['Recall']
            result_dict[f'{task_name}_Sp'] = val_metrics_per_task[task_id]['Specificity']
            result_dict[f'{task_name}_PR'] = val_metrics_per_task[task_id]['Precision']
            result_dict[f'{task_name}_F1'] = val_metrics_per_task[task_id]['F1']
            result_dict[f'{task_name}_ROC'] = val_metrics_per_task[task_id]['AUC_ROC']
            result_dict[f'{task_name}_MCC'] = val_metrics_per_task[task_id]['MCC']
            result_dict[f'{task_name}_PRC'] = val_metrics_per_task[task_id]['AUC_PRC']
        
        result_df = pd.DataFrame([result_dict])
        if epoch == 1:
            result_df.to_csv(args.result_dir, index=False, mode='w')
        else:
            result_df.to_csv(args.result_dir, index=False, mode='a', header=False)
        
        # 早停检查 - 使用验证损失
        early_stopping(avg_val_loss, model)  
        if early_stopping.early_stop:
            print("触发早停机制")
            break
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    # 打印最佳结果
    print("\n最佳指标:")
    print("-" * 40)
    print(f"整体最佳F1: {best_metrics['overall_F1']:.4f}")
    print(f"整体最佳AUC: {best_metrics['overall_AUC']:.4f}")
    
    print("\n各任务最佳F1分数:")
    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        print(f"{task_name}: {best_metrics['task_best_F1'][task_id]:.4f}")
    
    return best_metrics

if __name__ == "__main__":
    # 默认路径
    train_path = "./data/train_set.csv"
    val_path = "./data/valid_set.csv"
    my_seed = 42
    result_path = f"./results/multitask_esm2_results_seed{my_seed}.csv"
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="多任务学习细菌分泌系统效应蛋白分类(ESM-2)")
    
    # 数据参数
    parser.add_argument("--seed", type=int, default=my_seed, help="随机种子")
    parser.add_argument("--train_dir", type=str, default=train_path, help="训练数据路径")
    parser.add_argument("--val_dir", type=str, default=val_path, help="验证数据路径")
    parser.add_argument("--result_dir", type=str, default=result_path, help="结果保存路径")
    
    # 训练参数 - 针对ESM-2调整
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")  # ESM-2更大，可能需要更小的batch
    parser.add_argument("--frozen_layers", type=int, default=24, help="冻结的ESM层数")  # 默认冻结前24层
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率")  # ESM-2可能需要稍高的学习率
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    
    # 模型保存
    parser.add_argument("--checkpoint_path", type=str, 
                       default="./checkpoints/best_multitask_esm2_model.pt", 
                       help="模型保存路径")
    
    args = parser.parse_args()
    
    print("开始多任务学习训练(ESM-2)...")
    print(f"任务数量: {NUM_TASKS}")
    print(f"使用模型: ESM-2 (t33_650M_UR50D)")
    print(f"训练数据: {args.train_dir}")
    print(f"验证数据: {args.val_dir}")
    print(f"结果保存: {args.result_dir}")
    print(f"冻结层数: {args.frozen_layers}")
    
    best_metrics = train(args)