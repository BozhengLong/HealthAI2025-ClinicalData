# HealthAI - 医疗诊断模型

这个项目使用大型语言模型进行医疗诊断，通过微调Qwen2.5-7B-Instruct模型，使其能够根据病历信息给出准确的诊断和诊断依据。

## 项目结构

```
HealthAI/
├── data/                      # 数据目录
│   ├── camp_data_step_2_without_answer.jsonl  # 测试数据
│   └── train_labeled_by_qwen.jsonl            # 训练数据
├── models/                    # 模型目录
│   ├── qwen/                  # 基础模型
│   └── qwen_lora_finetuned/   # 微调后的模型
├── scripts/                   # 脚本目录
│   ├── finetune_qwen.py       # 微调脚本
│   ├── infer.py               # 原始推理脚本
│   ├── infer_json.py          # 改进的JSON格式推理脚本
│   ├── test_json_format.py    # JSON格式测试脚本
│   └── evaluate_json_parsing.py # 评估JSON解析成功率的脚本
└── README.md                  # 项目说明
```

## 环境设置

本项目需要在conda环境中运行，确保已安装以下依赖：

```bash
# 激活conda环境
conda activate healthai

# 安装依赖
pip install torch transformers peft datasets tqdm
```

## 使用说明

### 1. 微调模型

使用以下命令微调模型：

```bash
python scripts/finetune_qwen.py \
  --train_file ../data/train_labeled_by_qwen.jsonl \
  --model_name ../models/qwen/Qwen2___5-7B-Instruct \
  --output_dir ../models/qwen_lora_finetuned \
  --num_epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5
```

参数说明：
- `--train_file`: 训练数据文件路径
- `--model_name`: 基础模型路径
- `--output_dir`: 输出目录
- `--num_epochs`: 训练轮数
- `--batch_size`: 批处理大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--max_samples`: 用于测试的最大样本数（可选）

### 2. 测试JSON格式输出

使用以下命令测试模型的JSON格式输出：

```bash
python scripts/test_json_format.py \
  <model_base> \
  <adapter_dir> \
  [test_cases]
```

参数说明：
- `model_base`: 基础模型路径
- `adapter_dir`: LoRA适配器路径
- `test_cases`: 测试样本数量（可选，默认为3）

示例：
```bash
python scripts/test_json_format.py \
  ../models/qwen/Qwen2___5-7B-Instruct \
  ../models/qwen_lora_finetuned/20250316_001237/final_model \
  3
```

### 3. 使用改进的JSON格式推理

使用以下命令进行推理：

```bash
python scripts/infer_json.py \
  --model_base ../models/qwen/Qwen2___5-7B-Instruct \
  --adapter_dir ../models/qwen_lora_finetuned/20250316_001237/final_model \
  --data_path ../data/camp_data_step_2_without_answer.jsonl \
  --output_file results_json.jsonl \
  --save_interval 5
```

参数说明：
- `--model_base`: 基础模型路径
- `--adapter_dir`: LoRA适配器路径
- `--data_path`: 测试数据路径
- `--output_file`: 输出文件路径
- `--save_interval`: 保存结果的间隔（样本数）
- `--start_index`: 开始处理的样本索引（可选）
- `--end_index`: 结束处理的样本索引（可选）
- `--max_new_tokens`: 生成的最大token数（可选，默认256）
- `--temperature`: 生成温度（可选，默认0.7）
- `--top_p`: top-p采样参数（可选，默认0.8）
- `--top_k`: top-k采样参数（可选，默认20）

### 4. 评估JSON解析成功率

使用以下命令评估JSON解析成功率：

```bash
python scripts/evaluate_json_parsing.py \
  --results_file results_json.jsonl
```

参数说明：
- `--results_file`: 结果文件路径

## 内存优化

本项目在处理大型模型时进行了多项内存优化：

1. 使用LoRA进行参数高效微调
2. 使用bfloat16精度减少内存使用
3. 启用梯度检查点
4. 使用动态填充而非固定最大长度
5. 增量保存结果并定期清理内存
6. 优化PyTorch内存分配器减少内存碎片

## 输出格式

模型输出采用JSON格式，包含以下字段：

```json
{
  "diseases": "疾病名称，多个疾病用逗号分隔",
  "reason": "详细的诊断依据，包括关键症状和分析"
}
```

## 故障排除

如果遇到"CUDA out of memory"错误，可以尝试：

1. 减少批处理大小或梯度累积步数
2. 减少模型最大输入长度
3. 使用更小的LoRA配置（减小r和alpha值）
4. 使用更激进的内存优化设置

## 许可证

[MIT License](LICENSE) 