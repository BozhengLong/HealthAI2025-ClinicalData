#!/usr/bin/env python3
"""
HealthAI 2025 比赛 - 统一知识蒸馏引擎
整合多种蒸馏策略，专为HealthAI 2025临床医学数据处理竞赛设计

比赛信息：https://competition.pkucxpl.com/
- 支持API蒸馏和本地模型蒸馏
- 严格符合比赛规则：仅使用Qwen2.5-7B-Instruct作为最终模型
- 针对医疗诊断场景优化的权重调整策略
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch.nn.functional as F

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.error_handler import ErrorHandler
from .api_distillation import APIDistillationEngine, APIDistillationConfig

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """蒸馏配置"""
    # 模型配置
    teacher_model_path: Optional[str] = None
    student_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # 蒸馏策略
    distillation_type: str = "api"  # "api", "local", "hybrid"
    temperature: float = 4.0
    alpha: float = 0.7  # KL散度权重
    
    # 权重配置
    reasoning_weight: float = 3.0
    diagnosis_weight: float = 2.5
    feature_weight: float = 0.3
    
    # 训练配置
    max_length: int = 1024
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    # 输出配置
    output_dir: str = "outputs/distilled_model"
    save_steps: int = 500

class BaseDistillationStrategy(ABC):
    """蒸馏策略基类"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.error_handler = ErrorHandler()
    
    @abstractmethod
    def prepare_teacher_outputs(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """准备教师模型输出"""
        pass
    
    @abstractmethod
    def compute_distillation_loss(self, 
                                student_logits: torch.Tensor,
                                teacher_outputs: Dict[str, Any],
                                labels: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        pass

class APIDistillationStrategy(BaseDistillationStrategy):
    """API蒸馏策略"""
    
    def __init__(self, config: DistillationConfig, api_config: APIDistillationConfig):
        super().__init__(config)
        self.api_engine = APIDistillationEngine(api_config)
    
    def prepare_teacher_outputs(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """通过API获取教师模型输出"""
        outputs = []
        for content in inputs:
            result = self.api_engine.generate_diagnosis(content)
            if result['success']:
                outputs.append(result['result'])
            else:
                logger.warning(f"API调用失败: {result.get('error', 'Unknown error')}")
                outputs.append({
                    'reason': '分析失败',
                    'diseases': '未知',
                    'feature_content': content
                })
        return outputs
    
    def compute_distillation_loss(self, 
                                student_logits: torch.Tensor,
                                teacher_outputs: Dict[str, Any],
                                labels: torch.Tensor) -> torch.Tensor:
        """计算基于响应的蒸馏损失"""
        # 对于API蒸馏，主要使用标准的语言建模损失
        # 可以根据内容类型调整权重
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算基础损失
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        losses = loss_fct(flat_logits, flat_labels)
        
        # 根据内容类型调整权重（这里简化处理）
        weighted_losses = losses * self.config.reasoning_weight
        
        return weighted_losses.mean()

class LocalDistillationStrategy(BaseDistillationStrategy):
    """本地模型蒸馏策略"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__(config)
        self.teacher_model = None
        self.teacher_tokenizer = None
        self._load_teacher_model()
    
    def _load_teacher_model(self):
        """加载教师模型"""
        if self.config.teacher_model_path:
            try:
                self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.teacher_model_path,
                    trust_remote_code=True
                )
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    self.config.teacher_model_path,
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.teacher_model.eval()
                logger.info("教师模型加载成功")
            except Exception as e:
                self.error_handler.handle_error(e, "教师模型加载失败")
                raise
    
    def prepare_teacher_outputs(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """获取教师模型的logits输出"""
        if not self.teacher_model:
            raise ValueError("教师模型未加载")
        
        outputs = []
        with torch.no_grad():
            for text in inputs:
                tokens = self.teacher_tokenizer.encode(text, return_tensors="pt")
                teacher_outputs = self.teacher_model(tokens)
                outputs.append({
                    'logits': teacher_outputs.logits,
                    'tokens': tokens
                })
        return outputs
    
    def compute_distillation_loss(self, 
                                student_logits: torch.Tensor,
                                teacher_outputs: Dict[str, Any],
                                labels: torch.Tensor) -> torch.Tensor:
        """计算KL散度蒸馏损失"""
        teacher_logits = teacher_outputs['logits']
        
        # 计算KL散度损失
        student_probs = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        kl_loss *= (self.config.temperature ** 2)
        
        # 计算标准损失
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
        
        # 组合损失
        total_loss = self.config.alpha * kl_loss + (1 - self.config.alpha) * ce_loss
        return total_loss

class DistillationEngine:
    """统一蒸馏引擎"""
    
    def __init__(self, config: DistillationConfig, api_config: Optional[APIDistillationConfig] = None):
        self.config = config
        self.student_model = None
        self.student_tokenizer = None
        self.error_handler = ErrorHandler()
        
        # 选择蒸馏策略
        if config.distillation_type == "api":
            if not api_config:
                raise ValueError("API蒸馏需要提供api_config")
            self.strategy = APIDistillationStrategy(config, api_config)
        elif config.distillation_type == "local":
            self.strategy = LocalDistillationStrategy(config)
        else:
            raise ValueError(f"不支持的蒸馏类型: {config.distillation_type}")
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
    
    def load_student_model(self):
        """加载学生模型"""
        try:
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.student_tokenizer.pad_token_id is None:
                self.student_tokenizer.pad_token_id = self.student_tokenizer.eos_token_id
            
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            
            logger.info("学生模型加载成功")
            
        except Exception as e:
            self.error_handler.handle_error(e, "学生模型加载失败")
            raise
    
    def prepare_distillation_dataset(self, data_path: str) -> Dataset:
        """准备蒸馏数据集"""
        try:
            # 加载数据
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                else:
                    data_content = json.load(f)
                    if isinstance(data_content, dict) and 'data' in data_content:
                        data = data_content['data']
                    else:
                        data = data_content
            
            logger.info(f"加载蒸馏数据: {len(data)} 条")
            
            # 准备教师输出
            contents = [item.get('feature_content', '') for item in data]
            teacher_outputs = self.strategy.prepare_teacher_outputs(contents)
            
            # 构建训练样本
            training_samples = []
            for i, (item, teacher_output) in enumerate(zip(data, teacher_outputs)):
                if self.config.distillation_type == "api":
                    # API蒸馏：使用生成的完整响应
                    prompt = f"请根据患者信息进行医疗诊断分析：\n{item.get('feature_content', '')}\n\n诊断结果："
                    response = json.dumps(teacher_output, ensure_ascii=False, indent=2)
                    full_text = prompt + "\n" + response
                else:
                    # 本地蒸馏：使用原始格式
                    full_text = item.get('full_text', '')
                
                training_samples.append({
                    'full_text': full_text,
                    'teacher_output': teacher_output
                })
            
            # 转换为Dataset
            def tokenize_function(examples):
                tokenized = self.student_tokenizer(
                    examples['full_text'],
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_length,
                    return_tensors=None
                )
                tokenized['labels'] = tokenized['input_ids'].copy()
                tokenized['teacher_output'] = examples['teacher_output']
                return tokenized
            
            dataset = Dataset.from_list(training_samples)
            dataset = dataset.map(
                tokenize_function,
                remove_columns=['full_text']
            )
            
            logger.info(f"蒸馏数据集准备完成: {len(dataset)} 条")
            return dataset
            
        except Exception as e:
            self.error_handler.handle_error(e, "蒸馏数据准备失败")
            raise
    
    def run_distillation(self, data_path: str):
        """执行蒸馏训练"""
        logger.info("=== HealthAI 知识蒸馏开始 ===")
        
        # 1. 加载学生模型
        self.load_student_model()
        
        # 2. 准备蒸馏数据
        train_dataset = self.prepare_distillation_dataset(data_path)
        
        # 3. 创建自定义Trainer
        class DistillationTrainer(Trainer):
            def __init__(self, distillation_strategy, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.distillation_strategy = distillation_strategy
            
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                teacher_outputs = inputs.pop("teacher_output")
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 计算蒸馏损失
                loss = self.distillation_strategy.compute_distillation_loss(
                    logits, teacher_outputs, labels
                )
                
                return (loss, outputs) if return_outputs else loss
        
        # 4. 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            save_steps=self.config.save_steps,
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            report_to=None
        )
        
        # 5. 创建Trainer并训练
        trainer = DistillationTrainer(
            distillation_strategy=self.strategy,
            model=self.student_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.student_tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.student_tokenizer,
                mlm=False
            )
        )
        
        # 开始训练
        logger.info("开始蒸馏训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model(self.config.output_dir)
        self.student_tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"蒸馏训练完成，模型已保存到: {self.config.output_dir}")
        logger.info("=== HealthAI 知识蒸馏完成 ===")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HealthAI 知识蒸馏")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--distillation_type", type=str, default="api", 
                        choices=["api", "local"], help="蒸馏类型")
    parser.add_argument("--teacher_model", type=str, help="教师模型路径（本地蒸馏）")
    parser.add_argument("--student_model", type=str, 
                        default="Qwen/Qwen2.5-7B-Instruct", help="学生模型路径")
    parser.add_argument("--api_key", type=str, help="API密钥（API蒸馏）")
    parser.add_argument("--output_dir", type=str, 
                        default="outputs/distilled_model", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建蒸馏配置
    config = DistillationConfig(
        teacher_model_path=args.teacher_model,
        student_model_path=args.student_model,
        distillation_type=args.distillation_type,
        output_dir=args.output_dir
    )
    
    # API配置（如果需要）
    api_config = None
    if args.distillation_type == "api":
        if not args.api_key:
            raise ValueError("API蒸馏需要提供API密钥")
        api_config = APIDistillationConfig(api_key=args.api_key)
    
    # 创建蒸馏引擎并开始训练
    engine = DistillationEngine(config, api_config)
    engine.run_distillation(args.data_path)

if __name__ == "__main__":
    main() 