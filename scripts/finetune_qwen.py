## Author: Bozheng Long
## Date Created: 2025-03-15
## Last Modified: 2025-03-16
## Description: 使用 Qwen_2.5-72B-instruct 生成的结果对 Qwen_2.5-7B-instruct 模型进行微调

import json
import argparse
import logging
import math
import gc
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    default_data_collator,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split

# 导入工具函数
from utils import get_project_root, get_data_path, get_model_path, setup_logging, ensure_dir

# 优化PyTorch内存分配器，减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置日志
logger = setup_logging(log_file=Path(get_project_root()) / "logs" / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def load_data(file_path):
    """加载训练数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"训练数据文件不存在: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                # 数据验证
                required_fields = ['feature_content', 'diseases', 'reason']
                if not all(field in item for field in required_fields):
                    logger.warning(f"数据缺少必要字段: {item.get('id', 'unknown')}")
                    continue
                if not item['feature_content'].strip() or not item['diseases'].strip():
                    logger.warning(f"数据字段为空: {item.get('id', 'unknown')}")
                    continue
                # 数据清理
                item['feature_content'] = item['feature_content'].strip()
                item['diseases'] = item['diseases'].strip()
                item['reason'] = item['reason'].strip()
                data.append(item)
            except json.JSONDecodeError:
                logger.error(f"JSON解析错误，跳过该行")
                continue
    
    logger.info(f"成功加载 {len(data)} 条训练数据")
    return data

def generate_prompt(example):
    try:
        content = example['feature_content'].strip()
        diseases = example['diseases'].strip()
        reason = example['reason'].strip()
        
        # 使用更清晰的JSON格式的prompt模板
        prompt = (
            "你是一位经验丰富的医生，专门从事疾病诊断工作。请仔细阅读以下病历信息，"
            "根据患者的症状、体征和检查结果，给出准确的诊断和详细的诊断依据。\n\n"
            f"病历信息：\n{content}\n\n"
            "请以以下JSON格式输出您的诊断：\n"
            "{\n"
            '  "diseases": "简洁的疾病名称，多个疾病用逗号分隔",\n'
            '  "reason": "详细的诊断依据，包括关键症状和分析"\n'
            "}\n\n"
            "模型回答：\n"
            "{\n"
            f'  "diseases": "{diseases}",\n'
            f'  "reason": "{reason}"\n'
            "}\n"
        )
        return prompt
    except Exception as e:
        logger.error(f"生成prompt时出错: {str(e)}")
        return None

def preprocess_function(examples, tokenizer, max_length=2048):  # 减少最大长度
    prompts = []
    # 处理批量数据
    for i in range(len(examples['feature_content'])):
        example = {
            'feature_content': examples['feature_content'][i],
            'diseases': examples['diseases'][i],
            'reason': examples['reason'][i]
        }
        prompt = generate_prompt(example)
        if prompt:
            prompts.append(prompt)
        else:
            logger.warning(f"跳过无效样本: {i}")
    
    # 使用动态填充以节省内存
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    # 记录被截断的样本
    truncated = sum(len(ids) == max_length for ids in model_inputs['input_ids'])
    if truncated > 0:
        logger.warning(f"有 {truncated} 个样本被截断")
    
    # 对于因果语言模型，标签就是输入本身
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def compute_metrics(eval_preds):
    # 这里可以添加自定义的评估指标
    return {}

def main():
    parser = argparse.ArgumentParser(description="使用LoRA方法微调Qwen模型进行医疗诊断")
    parser.add_argument("--train_file", type=str, 
                        default=str(get_data_path("train_labeled_by_qwen.jsonl")),
                        help="训练数据文件路径")
    parser.add_argument("--output_dir", type=str, 
                        default=str(get_model_path("qwen_lora_finetuned")),
                        help="模型输出目录")
    parser.add_argument("--model_name", type=str, 
                        default=str(get_model_path("qwen/Qwen2___5-7B-Instruct")),
                        help="基础模型路径")
    parser.add_argument("--max_length", type=int, default=2048, 
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="保存检查点的步数间隔")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="用于测试的最大样本数，默认使用全部数据")
    args = parser.parse_args()

    # 创建输出目录
    ensure_dir(args.output_dir)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    ensure_dir(output_dir)

    try:
        # 载入数据
        data = load_data(args.train_file)
        
        # 如果指定了max_samples，只使用部分数据
        if args.max_samples is not None:
            if args.max_samples < len(data):
                data = data[:args.max_samples]
                logger.info(f"使用前 {args.max_samples} 条数据进行测试训练")
        
        # 不需要验证集，所以使用更多数据训练
        train_data = data
        train_dataset = Dataset.from_list(train_data)
        
        logger.info(f"训练集大小: {len(train_data)}")

        # 载入模型和分词器
        logger.info(f"开始加载模型: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        
        # 确保tokenizer有正确的padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 手动清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 加载模型 - 使用更激进的内存优化
        logger.info("使用优化后的内存设置加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,  # 使用bfloat16降低内存使用
            trust_remote_code=True,
            device_map=None
        )
        
        # 将模型移到GPU（如果有）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 确保模型知道padding token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # 准备模型进行训练
        model.gradient_checkpointing_enable()  # 启用梯度检查点
        model.enable_input_require_grads()  # 确保输入需要梯度
        model.config.use_cache = False  # 禁用KV缓存以支持梯度检查点
        
        # 配置 LoRA - 使用最小化配置
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # 只修改查询和值投影，减少一半的参数
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 获取PEFT模型
        model = get_peft_model(model, lora_config)
        
        # 将模型移到训练设备
        model = model.to(device)
        
        # 确保模型处于训练模式
        model.train()
        
        # 打印可训练参数
        model.print_trainable_parameters()

        # 数据预处理
        train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # 再次清理内存
        gc.collect()
        torch.cuda.empty_cache()

        # 训练配置 - 内存最优化
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=10,
            save_steps=args.save_steps,
            # 完全禁用评估
            evaluation_strategy="no",
            save_strategy="steps",
            save_total_limit=2,
            fp16=True,
            report_to="none",  # 禁用wandb
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            logging_first_step=True,
        )

        # 训练器配置 - 不使用评估数据集
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        )

        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 保存最终模型
        logger.info("保存最终模型...")
        model.save_pretrained(os.path.join(output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        
        logger.info(f"训练完成，模型保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()