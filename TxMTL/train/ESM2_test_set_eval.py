import argparse
from transformers import EsmModel, EsmTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, 
                            f1_score, roc_auc_score, matthews_corrcoef, 
                            average_precision_score, confusion_matrix)
import random
import os

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


# 导入你的多任务模型
from MTL_model import MultiTask_ESM_Classifier  # 修改导入

class MultiTaskDataset(Dataset):
    """多任务学习测试数据集"""
    def __init__(self, sequences, task_labels, tokenizer, max_len):
        """
        参数:
            sequences: 蛋白质序列列表 (带空格)
            task_labels: 任务标签列表 (如 'T1SE', 'T2SE'等)
            tokenizer: BERT tokenizer
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
        
        # Tokenize序列
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

def _get_test_data_loader(batch_size, test_dir):
    """创建测试数据加载器"""
    df = pd.read_csv(test_dir)
    
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
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader



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

def test(args):
    """测试函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载测试数据
    print("加载测试数据...")
    test_loader = _get_test_data_loader(args.batch_size, args.test_dir)
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 初始化模型
    print("初始化模型...")
    model = MultiTask_ESM_Classifier(num_tasks=NUM_TASKS, num_classes_per_task=2)  # 修改模型
    
    # 加载训练好的模型权重
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print(f"成功加载模型权重: {args.checkpoint_path}")
    else:
        raise FileNotFoundError(f"模型权重文件不存在: {args.checkpoint_path}")
    

    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 为每个任务存储预测结果
    test_task_predictions = {task_id: {'true': [], 'pred': [], 'probs': []} 
                            for task_id in range(NUM_TASKS)}
    
    # 存储每个样本的详细信息
    sample_details = []
    
    # 测试循环
    print("\n开始测试评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            all_task_labels = batch['all_task_labels'].to(device)
            true_task_ids = batch['true_task_id']  # 真实任务ID
            
            # 前向传播
            task_outputs = model(input_ids, attention_mask=attention_mask)
            
            # 收集每个任务的预测结果
            for task_id in range(NUM_TASKS):
                task_output = task_outputs[task_id]
                probs = torch.softmax(task_output, dim=1)[:, 1].cpu().numpy()  # 正类概率
                preds = (probs > 0.5).astype(int)  # 基于0.5阈值
                labels = all_task_labels[:, task_id].cpu().numpy()
                
                test_task_predictions[task_id]['true'].extend(labels)
                test_task_predictions[task_id]['pred'].extend(preds)
                test_task_predictions[task_id]['probs'].extend(probs)
            
            # 收集每个样本的详细信息
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                sample_info = {
                    'sequence_idx': batch_idx * args.batch_size + i,
                    'true_task': REVERSE_TASK_MAPPING.get(true_task_ids[i].item(), 'Unknown'),
                    'true_task_id': true_task_ids[i].item(),
                }
                
                # 添加每个任务的预测概率和预测结果
                for task_id in range(NUM_TASKS):
                    task_name = REVERSE_TASK_MAPPING[task_id]
                    prob = torch.softmax(task_outputs[task_id], dim=1)[i, 1].item()
                    pred = 1 if prob > 0.5 else 0
                    sample_info[f'{task_name}_prob'] = prob
                    sample_info[f'{task_name}_pred'] = pred
                
                sample_details.append(sample_info)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 批次")
    
    print("\n" + "="*60)
    print("测试完成，开始计算指标...")
    print("="*60)
    
    # 计算每个任务的指标
    test_metrics_per_task = []
    overall_test_metrics = {'ACC': [], 'Sn': [], 'Sp': [], 'PR': [], 'F1': [], 
                           'AUC_ROC': [], 'MCC': [], 'AUC_PRC': []}
    
    print("\n各任务测试指标:")
    print("-" * 80)
    print(f"{'任务':6s} | {'ACC':6s} | {'Sn':6s} | {'Sp':6s} | {'PR':6s} | {'F1':6s} | {'AUROC':7s} | {'MCC':6s} | {'AUPRC':7s}")
    print("-" * 80)
    
    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        y_true = np.array(test_task_predictions[task_id]['true'])
        y_pred = np.array(test_task_predictions[task_id]['pred'])
        y_probs = np.array(test_task_predictions[task_id]['probs'])
        
        # 计算该任务的指标
        task_metrics = calculate_metrics(y_true, y_pred, y_probs)
        test_metrics_per_task.append(task_metrics)
        
        # 收集整体指标
        overall_test_metrics['ACC'].append(task_metrics['ACC'])
        overall_test_metrics['Sn'].append(task_metrics['Recall'])
        overall_test_metrics['Sp'].append(task_metrics['Specificity'])
        overall_test_metrics['PR'].append(task_metrics['Precision'])
        overall_test_metrics['F1'].append(task_metrics['F1'])
        overall_test_metrics['AUC_ROC'].append(task_metrics['AUC_ROC'])
        overall_test_metrics['MCC'].append(task_metrics['MCC'])
        overall_test_metrics['AUC_PRC'].append(task_metrics['AUC_PRC'])
        
        # 打印该任务的指标
        print(f"{task_name:6s} | {task_metrics['ACC']:.4f} | {task_metrics['Recall']:.4f} | "
              f"{task_metrics['Specificity']:.4f} | {task_metrics['Precision']:.4f} | "
              f"{task_metrics['F1']:.4f} | {task_metrics['AUC_ROC']:.4f} | "
              f"{task_metrics['MCC']:.4f} | {task_metrics['AUC_PRC']:.4f}")
    
    # 计算整体平均指标
    print("-" * 80)
    overall_avg_ACC = np.mean(overall_test_metrics['ACC'])
    overall_avg_Sn = np.mean(overall_test_metrics['Sn'])
    overall_avg_Sp = np.mean(overall_test_metrics['Sp'])
    overall_avg_PR = np.mean(overall_test_metrics['PR'])
    overall_avg_F1 = np.mean(overall_test_metrics['F1'])
    overall_avg_ROC = np.mean(overall_test_metrics['AUC_ROC'])
    overall_avg_MCC = np.mean(overall_test_metrics['MCC'])
    overall_avg_PRC = np.mean(overall_test_metrics['AUC_PRC'])
    
    print(f"{'整体平均':6s} | {overall_avg_ACC:.4f} | {overall_avg_Sn:.4f} | "
          f"{overall_avg_Sp:.4f} | {overall_avg_PR:.4f} | "
          f"{overall_avg_F1:.4f} | {overall_avg_ROC:.4f} | "
          f"{overall_avg_MCC:.4f} | {overall_avg_PRC:.4f}")
    
    # 保存详细预测结果
    if args.save_predictions:
        predictions_df = pd.DataFrame(sample_details)
        predictions_path = args.result_dir.replace('.csv', '_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\n详细预测结果已保存到: {predictions_path}")
    
    # 准备结果字典
    result_dict = {
        'test_overall_ACC': overall_avg_ACC,
        'test_overall_Sn': overall_avg_Sn,
        'test_overall_Sp': overall_avg_Sp,
        'test_overall_PR': overall_avg_PR,
        'test_overall_F1': overall_avg_F1,
        'test_overall_AUROC': overall_avg_ROC,
        'test_overall_MCC': overall_avg_MCC,
        'test_overall_AUPRC': overall_avg_PRC,
    }
    
    # 添加每个任务的指标
    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        task_metrics = test_metrics_per_task[task_id]
        result_dict[f'test_{task_name}_ACC'] = task_metrics['ACC']
        result_dict[f'test_{task_name}_Sn'] = task_metrics['Recall']
        result_dict[f'test_{task_name}_Sp'] = task_metrics['Specificity']
        result_dict[f'test_{task_name}_PR'] = task_metrics['Precision']
        result_dict[f'test_{task_name}_F1'] = task_metrics['F1']
        result_dict[f'test_{task_name}_ROC'] = task_metrics['AUC_ROC']
        result_dict[f'test_{task_name}_MCC'] = task_metrics['MCC']
        result_dict[f'test_{task_name}_PRC'] = task_metrics['AUC_PRC']
    
    # 保存结果到CSV
    result_df = pd.DataFrame([result_dict])
    result_df.to_csv(args.result_dir, index=False)
    print(f"\n测试结果已保存到: {args.result_dir}")
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("测试汇总结果:")
    print("="*60)
    print(f"整体ACC: {overall_avg_ACC:.4f}")
    print(f"整体Sn(Recall): {overall_avg_Sn:.4f}")
    print(f"整体Sp(Specificity): {overall_avg_Sp:.4f}")
    print(f"整体PR(Precision): {overall_avg_PR:.4f}")
    print(f"整体F1: {overall_avg_F1:.4f}")
    print(f"整体AUROC: {overall_avg_ROC:.4f}")
    print(f"整体MCC: {overall_avg_MCC:.4f}")
    print(f"整体AUPRC: {overall_avg_PRC:.4f}")
    
    return result_dict

if __name__ == "__main__":
    # 默认路径
    test_path = "./data/test_set.csv"  # 测试集路径
    my_seed = 42
    result_path = f"./results/multitask_test_results_seed{my_seed}.csv"
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="多任务学习细菌分泌系统效应蛋白分类 - 测试评估")
    
    # 数据参数
    parser.add_argument("--seed", type=int, default=my_seed, help="随机种子")
    parser.add_argument("--test_dir", type=str, default=test_path, help="测试数据路径")
    parser.add_argument("--result_dir", type=str, default=result_path, help="结果保存路径")
    
    # 模型参数
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小（测试时建议为1）")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="./checkpoints/best_multitask_esm2_model.pt", 
                       help="训练好的模型权重路径")
    
    # 其他参数
    parser.add_argument("--save_predictions", action="store_true", 
                       default=True, help="是否保存详细预测结果")
    
    args = parser.parse_args()
    
    print("开始多任务学习测试评估...")
    print(f"任务数量: {NUM_TASKS}")
    print(f"测试数据: {args.test_dir}")
    print(f"模型权重: {args.checkpoint_path}")
    print(f"结果保存: {args.result_dir}")
    print(f"批次大小: {args.batch_size}")
    
    results = test(args)
    print("\n测试完成！")