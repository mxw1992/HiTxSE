import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import os
import sys
from transformers import EsmTokenizer

# 如果你本地有 fasta2csv 模块，可以保留；没有也没关系，会自动走内置 FASTA 解析
try:
    import fasta2csv
except ImportError:
    print("警告: 未找到 fasta2csv 模块，将使用内置的 FASTA 解析功能")

# =========================
# 配置参数（与训练代码保持一致）
# =========================
MAX_LEN = 2000
NUM_TASKS = 7
TASK_MAPPING = {
    'T1SE': 0, 'T2SE': 1, 'T3SE': 2, 'T4SE': 3,
    'T5SE': 4, 'T6SE': 5, 'T7SE': 6
}
REVERSE_TASK_MAPPING = {v: k for k, v in TASK_MAPPING.items()}

# 与训练代码保持一致：使用 ESM-2 tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# 与训练代码保持一致：导入 ESM 多任务模型
from MTL_model import MultiTask_ESM_Classifier


# =========================
# FASTA 解析
# =========================
def simple_fasta_parser(fasta_path):
    """简单的 FASTA 文件解析器"""
    sequences = []
    sequence_ids = []

    with open(fasta_path, 'r') as f:
        current_seq = ""
        current_id = ""

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_id and current_seq:
                    sequences.append(current_seq)
                    sequence_ids.append(current_id)
                current_id = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line

        if current_id and current_seq:
            sequences.append(current_seq)
            sequence_ids.append(current_id)

    return sequence_ids, sequences


def fasta_to_dataframe(fasta_path):
    """将 FASTA 文件转换为 DataFrame"""
    try:
        seq_name, csv_path = fasta2csv.fasta2csv(fasta_path)
        df = pd.read_csv(csv_path)

        if 'SEQUENCE_space' not in df.columns:
            if 'SEQUENCE' in df.columns:
                df['SEQUENCE_space'] = df['SEQUENCE'].apply(lambda x: ' '.join(str(x)))
            else:
                raise ValueError("生成的 CSV 缺少 SEQUENCE_space 或 SEQUENCE 列")

        if 'ID' not in df.columns:
            df['ID'] = [f"Seq_{i+1}" for i in range(len(df))]

        if 'SEQUENCE' not in df.columns:
            df['SEQUENCE'] = df['SEQUENCE_space'].str.replace(' ', '', regex=False)

        return df

    except Exception:
        print("使用内置 FASTA 解析器...")
        sequence_ids, sequences = simple_fasta_parser(fasta_path)

        # 与训练代码风格保持一致，构造带空格序列列
        sequences_with_spaces = [' '.join(seq) for seq in sequences]

        df = pd.DataFrame({
            'ID': sequence_ids,
            'SEQUENCE': sequences,
            'SEQUENCE_space': sequences_with_spaces
        })

        csv_path = fasta_path.replace('.fasta', '.csv').replace('.fa', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"FASTA 已转换为 CSV 并保存到: {csv_path}")

        return df


# =========================
# 数据集
# =========================
class MultiTaskSeqDataset(Dataset):
    """多任务学习未知序列预测数据集（ESM 版本）"""
    def __init__(self, sequences, tokenizer, max_len):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])

        encoding = self.tokenizer.encode_plus(
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
        }


def get_prediction_data_loader(batch_size, data_df):
    dataset = MultiTaskSeqDataset(
        sequences=data_df['SEQUENCE_space'].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return dataloader


# =========================
# 冻结 ESM 层（与训练代码一致）
# =========================
def freeze_esm_layers(model, num_frozen_layers):
    """冻结 ESM 的前几层"""
    if num_frozen_layers > 0:
        for layer in model.esm.encoder.layer[:num_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"冻结了 {num_frozen_layers} 层 ESM 参数")


# =========================
# 未知序列预测
# =========================
def predict_unknown_sequences(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 读取输入文件
    if args.test_dir.endswith('.fasta') or args.test_dir.endswith('.fa'):
        print(f"读取 FASTA 文件: {args.test_dir}")
        data_df = fasta_to_dataframe(args.test_dir)
    else:
        print(f"读取 CSV 文件: {args.test_dir}")
        data_df = pd.read_csv(args.test_dir)

        # 检查序列列
        if 'SEQUENCE_space' not in data_df.columns:
            if 'SEQUENCE' in data_df.columns:
                print("检测到 SEQUENCE 列，自动添加空格生成 SEQUENCE_space...")
                data_df['SEQUENCE_space'] = data_df['SEQUENCE'].apply(lambda x: ' '.join(str(x)))
            else:
                raise ValueError("CSV 文件需要包含 'SEQUENCE_space' 或 'SEQUENCE' 列")

        if 'SEQUENCE' not in data_df.columns:
            data_df['SEQUENCE'] = data_df['SEQUENCE_space'].str.replace(' ', '', regex=False)

        if 'ID' not in data_df.columns:
            data_df['ID'] = [f"Seq_{i+1}" for i in range(len(data_df))]

    print(f"共读取 {len(data_df)} 条序列")

    # 初始化模型
    print("初始化模型...")
    model = MultiTask_ESM_Classifier(num_tasks=NUM_TASKS, num_classes_per_task=2)

    # 加载权重
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print(f"成功加载模型权重: {args.checkpoint_path}")
    else:
        raise FileNotFoundError(f"模型权重文件不存在: {args.checkpoint_path}")

    # 与训练时一致时可保留，不一致也不会影响预测结果，只是冻结对 eval 基本没意义
    freeze_esm_layers(model, args.frozen_layers)

    model = model.to(device)
    model.eval()

    test_loader = get_prediction_data_loader(args.batch_size, data_df)

    all_task_probs = {task_id: [] for task_id in range(NUM_TASKS)}
    all_task_preds = {task_id: [] for task_id in range(NUM_TASKS)}

    print("\n开始预测...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            task_outputs = model(input_ids, attention_mask=attention_mask)

            for task_id in range(NUM_TASKS):
                task_output = task_outputs[task_id]   # [batch_size, 2]
                probs = torch.softmax(task_output, dim=1)[:, 1].cpu().numpy()
                preds = (probs > args.threshold).astype(int)

                all_task_probs[task_id].extend(probs.tolist())
                all_task_preds[task_id].extend(preds.tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 批次")

    print("\n" + "=" * 60)
    print("预测完成，生成结果...")
    print("=" * 60)

    # 结果表
    result_data = {
        'ID': data_df['ID'].tolist(),
        'Sequence': data_df['SEQUENCE'].tolist(),
    }

    # 每个任务输出概率、Yes/No、0/1
    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        result_data[f'{task_name}_Prob'] = all_task_probs[task_id]
        result_data[f'{task_name}_Pred'] = ['Yes' if x == 1 else 'No' for x in all_task_preds[task_id]]
        result_data[f'{task_name}_Binary'] = all_task_preds[task_id]

    # 综合预测：取最大概率任务
    prob_matrix = np.column_stack([all_task_probs[task_id] for task_id in range(NUM_TASKS)])
    predicted_task_indices = np.argmax(prob_matrix, axis=1)

    result_data['Predicted_Task'] = [REVERSE_TASK_MAPPING[idx] for idx in predicted_task_indices]
    result_data['Max_Prob'] = np.max(prob_matrix, axis=1)

    # 判断是否多个任务同时 > threshold
    pred_matrix = np.column_stack([all_task_preds[task_id] for task_id in range(NUM_TASKS)])
    multiple_positives = np.sum(pred_matrix, axis=1) > 1
    result_data['Multiple_Positives'] = ['Yes' if x else 'No' for x in multiple_positives]

    result_df = pd.DataFrame(result_data)

    # 打印统计信息
    print("\n预测统计信息:")
    print("-" * 40)
    total_sequences = len(result_df)
    print(f"总序列数: {total_sequences}")

    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        positive_count = sum(all_task_preds[task_id])
        positive_percent = (positive_count / total_sequences * 100) if total_sequences > 0 else 0.0
        print(f"{task_name}: {positive_count} 条序列预测为阳性 ({positive_percent:.1f}%)")

    print("\n综合任务预测分布:")
    predicted_counts = result_df['Predicted_Task'].value_counts()
    for task, count in predicted_counts.items():
        percent = count / total_sequences * 100
        print(f"  {task}: {count} 条序列 ({percent:.1f}%)")

    multiple_pos_count = np.sum(multiple_positives)
    print(f"\n多任务阳性序列数: {multiple_pos_count} ({multiple_pos_count / total_sequences * 100:.1f}%)")

    # 保存结果
    result_df.to_csv(args.result_dir, index=False)
    print(f"\n预测结果已保存到: {args.result_dir}")

    # 可选：保存详细结果
    if args.save_detailed:
        detailed_path = args.result_dir.replace('.csv', '_detailed.csv')
        detailed_df = result_df.copy()
        detailed_df.to_csv(detailed_path, index=False)
        print(f"详细预测结果已保存到: {detailed_path}")

    return result_df


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    test_path = "./data/unknown_sequences.fasta"   # 支持 FASTA 或 CSV
    result_path = "./results/multitask_esm2_predictions.csv"

    os.makedirs("./results", exist_ok=True)

    parser = argparse.ArgumentParser(description="多任务学习细菌分泌系统效应蛋白分类（ESM-2）- 未知序列预测")

    # 数据参数
    parser.add_argument("--test_dir", type=str, default=test_path,
                        help="测试数据路径（支持 FASTA 或 CSV）")
    parser.add_argument("--result_dir", type=str, default=result_path,
                        help="预测结果保存路径")

    # 模型参数
    parser.add_argument("--batch-size", type=int, default=1,
                        help="批次大小（预测时建议 1）")
    parser.add_argument("--frozen_layers", type=int, default=24,
                        help="冻结的 ESM 层数（建议与训练保持一致）")
    parser.add_argument("--checkpoint_path", type=str,
                        default="./checkpoints/best_multitask_esm2_model.pt",
                        help="训练好的模型权重路径")

    # 预测参数
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="阳性预测阈值")
    parser.add_argument("--save_detailed", action="store_true", default=False,
                        help="是否保存详细预测结果")

    args = parser.parse_args()

    print("开始多任务学习未知序列预测（ESM-2）...")
    print(f"任务数量: {NUM_TASKS}")
    print(f"输入文件: {args.test_dir}")
    print(f"模型权重: {args.checkpoint_path}")
    print(f"结果保存: {args.result_dir}")
    print(f"预测阈值: {args.threshold}")
    print(f"批次大小: {args.batch_size}")
    print(f"冻结层数: {args.frozen_layers}")

    if not os.path.exists(args.test_dir):
        print(f"错误: 输入文件不存在: {args.test_dir}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 模型权重文件不存在: {args.checkpoint_path}")
        sys.exit(1)

    results = predict_unknown_sequences(args)

    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)
    print("\n结果文件包含以下列：")
    print("1. ID: 序列标识符")
    print("2. Sequence: 原始蛋白质序列")
    for task_id in range(NUM_TASKS):
        task_name = REVERSE_TASK_MAPPING[task_id]
        print(f"3. {task_name}_Prob: {task_name} 的预测概率")
        print(f"4. {task_name}_Pred: {task_name} 的预测结果（Yes/No）")
        print(f"5. {task_name}_Binary: {task_name} 的二进制预测（0/1）")
    print("6. Predicted_Task: 综合预测任务（最大概率对应任务）")
    print("7. Max_Prob: 最大预测概率")
    print("8. Multiple_Positives: 是否有多个任务同时预测为阳性")