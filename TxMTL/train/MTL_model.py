from transformers import EsmModel, EsmTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiTask_ESM_Classifier(nn.Module):
    def __init__(self, num_tasks=7, num_classes_per_task=2):
        """
        参数:
            num_tasks: 任务数量 (T1SE-T7SE共7个)
            num_classes_per_task: 每个任务的类别数 (都是二分类，所以是2)
        """
        super(MultiTask_ESM_Classifier, self).__init__()
        
        # 共享的ESM-2 backbone

        self.esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        hidden_size = self.esm.config.hidden_size
        

        
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
#            nn.Dropout(p=0.1)  # 添加dropout防止过拟合
            nn.Linear(hidden_size, 640),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.2)
        )
        
        # 为每个任务创建独立的输出头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(640, 320),
                nn.LeakyReLU(inplace=False),
                nn.Dropout(p=0.2),
                nn.Linear(320, num_classes_per_task)
            ) for _ in range(num_tasks)
        ])
        
        self.num_tasks = num_tasks
        
    def mean_pooling(self, token_embeddings, attention_mask):
        """平均池化：使用attention_mask忽略padding tokens"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        # 获取ESM输出
        esm_output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一层隐藏状态
        last_hidden_state = esm_output.last_hidden_state
        
        # 平均池化
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        

        
        # 共享特征提取
        shared_features = self.shared_layers(pooled_output)
        
        # 每个任务独立的输出
        outputs = []
        for task_head in self.task_heads:
            outputs.append(task_head(shared_features))
        
        # 返回一个列表，包含每个任务的logits
        return outputs