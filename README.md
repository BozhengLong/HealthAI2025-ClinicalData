# HealthAI - 医疗诊断系统

![Python](https://img.shields.io/badge/Python-3.10.16-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![License](https://img.shields.io/badge/License-MIT-green)

HealthAI是一个基于大型语言模型的医疗诊断系统，通过对Qwen2.5-7B-Instruct模型进行参数高效微调（LoRA），使其能够根据病历信息给出准确的诊断和详细的诊断依据。系统采用JSON格式输出，便于后续处理和分析。

## 功能特点

- **高效微调**：使用LoRA技术进行参数高效微调，大幅降低计算资源需求
- **结构化输出**：采用JSON格式输出诊断结果，便于解析和分析
- **内存优化**：多项内存优化技术，支持在有限资源环境下运行
- **增量处理**：支持大规模数据的增量处理和结果保存
- **错误恢复**：完善的错误处理和日志记录，支持从错误中恢复

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
│   ├── infer.py               # 推理脚本
│   ├── generate_qwen_diagonosis.py # 使用Qwen-72B生成训练数据
│   ├── baseline.py            # 基准测试脚本
│   └── utils.py               # 工具函数模块
├── logs/                      # 日志目录
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

## 安装指南

### 环境要求

### 硬件配置
- GPU: NVIDIA RTX 4090D (24GB) * 1
- CPU: 16 vCPU Intel(R) Xeon(R) Platinum 8481C
- RAM: 32GB+

### 软件要求
- Python 3.10.16
- CUDA 11.8
- PyTorch 2.5.1
- 运行环境: AutoDL云服务器

### 预计运行时间
- `generate_qwen_diagonosis.py`: ~8小时
- `infer.py`: ~8小时

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/BozhengLong/HealthAI2025-ClinicalData.git
cd HealthAI2025-ClinicalData
```

2. 创建并激活conda环境
```bash
conda create -n healthai python=3.10.16
conda activate healthai
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 准备模型和数据
```bash
# 下载基础模型 (如果尚未下载)
# 将训练和测试数据放入data目录
```

## 使用指南

### 1. 数据准备

使用Qwen-72B模型生成训练数据：

```bash
# 从项目根目录运行
python scripts/generate_qwen_diagonosis.py
```

### 2. 模型微调

```bash
# 从项目根目录运行
python scripts/finetune_qwen.py \
  --train_file data/train_labeled_by_qwen.jsonl \
  --model_name models/qwen/Qwen2___5-7B-Instruct \
  --output_dir models/qwen_lora_finetuned \
  --num_epochs 3 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5
```

主要参数说明：
- `--train_file`: 训练数据文件路径
- `--model_name`: 基础模型路径
- `--output_dir`: 输出目录
- `--num_epochs`: 训练轮数
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--max_samples`: 用于测试的最大样本数（可选）

### 3. 模型推理

```bash
# 使用微调后的模型进行推理
python scripts/infer.py \
  --model_base models/qwen/Qwen2___5-7B-Instruct \
  --adapter_dir models/qwen_lora_finetuned/[时间戳]/final_model \
  --input_path data/camp_data_step_2_without_answer.jsonl \
  --output_path data/predict_output.jsonl \
  --save_interval 100
```

主要参数说明：
- `--model_base`: 基础模型路径
- `--adapter_dir`: LoRA适配器路径
- `--input_path`: 测试数据路径
- `--output_path`: 输出文件路径
- `--save_interval`: 保存结果的间隔（样本数）

## 技术细节

### 微调方法

本项目使用LoRA (Low-Rank Adaptation) 技术对大型语言模型进行参数高效微调。LoRA通过在原始模型权重旁边添加小型可训练的低秩矩阵，显著减少了可训练参数的数量，从而降低了计算和内存需求。

### 内存优化

为了在有限的GPU资源下运行大型模型，项目采用了多种内存优化技术：

1. **混合精度训练**：使用bfloat16精度减少内存使用
2. **梯度检查点**：在前向传播中只保存部分中间激活值，在反向传播时重新计算
3. **动态填充**：避免使用固定最大长度，减少填充token
4. **增量处理**：分批处理数据并定期保存结果
5. **内存回收**：定期执行垃圾回收和CUDA缓存清理

### 输出格式

模型输出采用JSON格式，包含以下字段：

```json
{
  "diseases": "疾病名称，多个疾病用逗号分隔",
  "reason": "详细的诊断依据，包括关键症状和分析"
}
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少批处理大小或梯度累积步数
   - 减少模型最大输入长度
   - 使用更小的LoRA配置（减小r和alpha值）

2. **训练过程中断**
   - 检查日志文件了解详细错误信息
   - 确保数据格式正确
   - 尝试减少训练样本数量进行测试

3. **JSON解析问题**
   - 检查生成的输出格式
   - 调整prompt模板以引导模型生成更规范的JSON
   - 优化解析策略，增强容错能力

## 贡献指南

欢迎对本项目进行贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个Pull Request

## 许可证

本项目采用MIT许可证 - 详情请参见[LICENSE](LICENSE)文件

## 联系方式

Bozheng Long - im.bzlong@gmail.com

项目链接: [https://github.com/BozhengLong/HealthAI2025-ClinicalData](https://github.com/BozhengLong/HealthAI2025-ClinicalData) 