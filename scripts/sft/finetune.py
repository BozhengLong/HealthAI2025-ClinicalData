#!/usr/bin/env python3
"""
HealthAI 2025 比赛 - 统一微调模块
专为HealthAI 2025临床医学数据处理竞赛设计的模型微调方案

比赛信息：https://competition.pkucxpl.com/
- 基础模型：Qwen2.5-7B-Instruct (比赛指定模型)
- 微调策略：LoRA + 4bit量化 + 梯度累积
- 应用场景：医疗诊断、临床推理、病历分析
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, 
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset, load_dataset
import torch.distributed as dist

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config_manager import ConfigManager
from core.model_loader import ModelLoader
from core.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """微调配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 1024
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 训练配置
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    save_steps: int = 500
    
    # 量化配置
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    
    # 输出配置
    output_dir: str = "outputs/sft_model"
    logging_steps: int = 10

class HealthAIFineTuner:
    """HealthAI微调器"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.error_handler = ErrorHandler()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
    
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 量化配置
            if self.config.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=True
                )
            else:
                bnb_config = None
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # 准备LoRA
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # 配置LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("模型和tokenizer加载成功")
            
        except Exception as e:
            self.error_handler.handle_error(e, "模型加载失败")
            raise
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """准备训练数据"""
        try:
            # 加载数据
            if data_path.endswith('.jsonl'):
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data_content = json.load(f)
                    if isinstance(data_content, dict) and 'data' in data_content:
                        data = data_content['data']
                    else:
                        data = data_content
            else:
                raise ValueError(f"不支持的数据格式: {data_path}")
            
            logger.info(f"加载数据: {len(data)} 条")
            
            def tokenize_function(examples):
                """tokenize函数"""
                # 构建训练文本
                if isinstance(examples, list):
                    texts = []
                    for item in examples:
                        if 'full_text' in item:
                            texts.append(item['full_text'])
                        elif 'prompt' in item and 'response' in item:
                            texts.append(item['prompt'] + '\n' + item['response'])
                        else:
                            # 构建基本格式
                            content = item.get('feature_content', '')
                            prompt = f"请根据患者信息进行医疗诊断分析：\n{content}\n\n诊断结果："
                            response = json.dumps(item.get('structured_result', {}), 
                                                ensure_ascii=False, indent=2)
                            texts.append(prompt + '\n' + response)
                else:
                    # 单个样本
                    if 'full_text' in examples:
                        texts = [examples['full_text']]
                    else:
                        content = examples.get('feature_content', '')
                        prompt = f"请根据患者信息进行医疗诊断分析：\n{content}\n\n诊断结果："
                        response = json.dumps(examples.get('structured_result', {}), 
                                            ensure_ascii=False, indent=2)
                        texts = [prompt + '\n' + response]
                
                # tokenize
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_length,
                    return_tensors=None
                )
                
                # 设置labels
                tokenized['labels'] = tokenized['input_ids'].copy()
                
                return tokenized
            
            # 转换为Dataset
            dataset = Dataset.from_list(data)
            dataset = dataset.map(
                lambda x: tokenize_function(x),
                batched=False,
                remove_columns=dataset.column_names
            )
            
            logger.info(f"数据集准备完成: {len(dataset)} 条")
            return dataset
            
        except Exception as e:
            self.error_handler.handle_error(e, "数据准备失败")
            raise
    
    def train(self, train_dataset: Dataset):
        """执行训练"""
        try:
            # 数据收集器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=self.config.warmup_steps,
                learning_rate=self.config.learning_rate,
                fp16=True,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                save_total_limit=2,
                remove_unused_columns=False,
                dataloader_drop_last=True,
                report_to=None
            )
            
            # 创建Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # 开始训练
            logger.info("开始微调训练...")
            trainer.train()
            
            # 保存模型
            trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info(f"训练完成，模型已保存到: {self.config.output_dir}")
            
        except Exception as e:
            self.error_handler.handle_error(e, "训练过程失败")
            raise
    
    def run_full_training(self, data_path: str):
        """完整训练流程"""
        logger.info("=== HealthAI 模型微调开始 ===")
        
        # 1. 加载模型
        self.load_model_and_tokenizer()
        
        # 2. 准备数据
        train_dataset = self.prepare_dataset(data_path)
        
        # 3. 执行训练
        self.train(train_dataset)
        
        logger.info("=== HealthAI 模型微调完成 ===")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HealthAI 模型微调")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--model_name", type=str, 
                        default="Qwen/Qwen2.5-7B-Instruct", help="基础模型名称")
    parser.add_argument("--output_dir", type=str, 
                        default="outputs/sft_model", help="输出目录")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    
    args = parser.parse_args()
    
    # 创建配置
    config = FineTuningConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # 创建微调器并开始训练
    finetuner = HealthAIFineTuner(config)
    finetuner.run_full_training(args.data_path)

if __name__ == "__main__":
    main() 