#!/usr/bin/env python3
"""
模型工具函数
提供模型加载、信息获取等通用功能
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def load_model_safely(model_path: str, 
                     device_map: str = "auto",
                     trust_remote_code: bool = True) -> Tuple[Any, Any]:
    """安全加载模型和tokenizer"""
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            padding_side="right"
        )
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map
        )
        
        logger.info(f"模型加载成功: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def get_model_info(model_path: str) -> Dict[str, Any]:
    """获取模型信息"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        info = {
            'model_path': model_path,
            'vocab_size': tokenizer.vocab_size,
            'model_max_length': tokenizer.model_max_length,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token_id': tokenizer.bos_token_id
        }
        
        return info
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        return {}

def estimate_model_memory(model_path: str) -> Dict[str, float]:
    """估算模型内存使用"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map="cpu"
        )
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算内存（假设float32）
        memory_gb = total_params * 4 / (1024**3)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_memory_gb': memory_gb
        }
        
    except Exception as e:
        logger.error(f"内存估算失败: {e}")
        return {}

def check_gpu_availability() -> Dict[str, Any]:
    """检查GPU可用性"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'gpu_names': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info['gpu_names'].append(torch.cuda.get_device_name(i))
    
    return gpu_info 