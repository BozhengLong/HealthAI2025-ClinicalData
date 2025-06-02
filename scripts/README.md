# HealthAI 2025 临床医学数据处理竞赛方案

## 🏆 比赛背景

本项目是 **[HealthAI 2025 健康智能挑战赛中的HealthAI 2025 临床医学数据处理竞赛](https://competition.pkucxpl.com/)** 的参赛方案。

### 比赛介绍

随着科技的快速进步和数字经济的蓬勃发展，我国医疗健康产业正面临着前所未有的机遇与挑战。以人工智能为代表的新一代信息技术正在帮助临床诊疗、药物研发等加速进步，展现出巨大的发展潜力和应用前景。

本次比赛由支付宝携手组委会提供数据集，涵盖医疗领域书籍、论文、网页、虚拟患者病历、临床报告等多种类型数据。

### 比赛规则与限制

- **指定模型**：必须使用 `Qwen2.5-7B-Instruct` 作为最终提交模型
- **允许操作**：预训练、SFT（监督微调）、Prompt Learning等
- **API限制**：禁止使用任何收费API进行数据蒸馏
- **开源API**：可以使用开源模型API（如DeepSeek、Qwen-72B等）

### 技术方案概述

本参赛方案采用 **两阶段技术路线**：

**阶段一：SFT微调**
1. 使用Qwen2.5-72B-Instruct API对训练数据集(`camp_data_step_1_without_answer.jsonl`)生成答案
2. 用生成的问答数据微调Qwen2.5-7B-Instruct模型
3. 使用微调后的模型对测试数据集(`camp_data_step_2_without_answer.jsonl`)生成答案

**阶段二：知识蒸馏（可选优化）**
1. 重新使用Qwen2.5-72B-Instruct API生成更高质量的soft label答案
2. 基于新的蒸馏数据进一步优化模型性能
3. 注：由于使用API形式的teacher模型，这是一种"伪蒸馏"方法

## 🏗️ 项目结构

```
scripts/
├── core/                           # 核心基础模块
│   ├── __init__.py
│   ├── config_manager.py           # 配置管理
│   ├── model_loader.py             # 模型加载器
│   └── error_handler.py            # 错误处理
├── sft/                            # 模型微调模块
│   ├── __init__.py
│   ├── finetune.py                 # 统一微调入口
│   ├── generate_diagnosis.py       # 诊断生成
│   └── inference.py                # 推理功能
├── distillation/                   # 知识蒸馏模块  
│   ├── __init__.py
│   ├── distillation_engine.py      # 统一蒸馏引擎
│   └── api_distillation.py         # API蒸馏方案
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── data_utils.py               # 数据处理工具
│   └── model_utils.py              # 模型工具
├── configs/                        # 配置文件
│   ├── sft_config.json             # SFT配置
│   ├── distillation_config.json    # 蒸馏配置
│   └── ds_config.json              # DeepSpeed配置
└── README.md                       # 项目文档
```

## 🚀 使用方法

### 环境准备

```bash
pip install -r requirements.txt
```

### 阶段一：SFT微调流程

#### 1. API生成训练数据

使用Qwen2.5-72B-Instruct API为训练数据生成答案：

```bash
python scripts/distillation/api_distillation.py \
    --api_key YOUR_QWEN_API_KEY \
    --input_file data/camp_data_step_1_without_answer.jsonl \
    --output_file data/sft_training_data.json \
    --teacher_model qwen2.5-72b-instruct \
    --max_samples 1000
```

#### 2. 模型微调

基于生成的问答数据微调Qwen2.5-7B-Instruct：

```bash
python scripts/sft/finetune.py \
    --data_path data/sft_training_data.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir models/sft_model \
    --num_epochs 3 \
    --batch_size 4
```

#### 3. 测试数据推理

使用微调后的模型对测试数据生成答案：

```bash
python scripts/sft/inference.py \
    --model_path models/sft_model \
    --input_file data/camp_data_step_2_without_answer.jsonl \
    --output_file results/test_results.json
```

### 阶段二：知识蒸馏优化（可选）

#### 1. 生成蒸馏数据

重新使用API生成更高质量的soft label数据：

```bash
python scripts/distillation/api_distillation.py \
    --api_key YOUR_QWEN_API_KEY \
    --input_file data/camp_data_step_1_without_answer.jsonl \
    --output_file data/distillation_training_data.json \
    --teacher_model qwen2.5-72b-instruct \
    --temperature 0.7 \
    --max_tokens 800
```

#### 2. 蒸馏训练

基于蒸馏数据进一步优化模型：

```bash
python scripts/distillation/distillation_engine.py \
    --data_path data/distillation_training_data.json \
    --distillation_type api \
    --api_key YOUR_QWEN_API_KEY \
    --student_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir models/distilled_model
```

## 📊 核心功能模块

### 🔧 SFT微调模块
- **LoRA微调**：高效参数更新，减少内存占用
- **4bit量化**：支持有限硬件资源下的模型训练
- **梯度累积**：优化小批次训练效果
- **多格式支持**：支持JSON和JSONL格式数据

### 🧠 知识蒸馏模块
- **API蒸馏**：通过API调用大模型进行"伪蒸馏"
- **权重优化**：针对医疗推理的token权重调整
- **温度控制**：调节soft label的软化程度
- **质量控制**：API调用失败处理和数据验证

### 🛠️ 工具支持
- **数据处理**：医疗数据预处理和格式转换
- **模型工具**：模型加载、信息获取、内存估算
- **错误处理**：完善的异常处理机制
- **配置管理**：统一的配置管理系统

## 🚨 注意事项

1. **API密钥安全**：请妥善保管API密钥，不要提交到代码仓库
2. **内存管理**：大模型训练需要足够的GPU内存，建议使用量化
3. **数据质量**：确保训练数据的质量和格式正确性
4. **两阶段独立**：SFT和蒸馏可以独立进行，蒸馏是可选的优化步骤

---

**HealthAI 2025 参赛方案** - 两阶段医疗AI模型优化技术路线 🏆🏥 