#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试JSON格式的提示和解析是否有效
"""

import json
import os
import sys
import logging
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_json(text):
    """从文本中提取JSON对象"""
    # 尝试找到JSON对象的开始和结束位置
    json_start = text.find('{')
    json_end = text.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = text[json_start:json_end]
        try:
            # 尝试解析JSON
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            # 如果解析失败，尝试修复常见的JSON格式问题
            try:
                # 替换多余的引号和转义字符
                fixed_json = re.sub(r'\\+', r'\\', json_str)
                # 尝试再次解析
                result = json.loads(fixed_json)
                return result
            except:
                return None
    return None

def test_json_format(model_base, adapter_dir, test_cases=3):
    """测试JSON格式的提示和解析"""
    # 加载模型和分词器
    logger.info(f"加载基础模型: {model_base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_base,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    logger.info(f"加载 LoRA 适配器: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 加载测试数据
    test_data_path = "../data/camp_data_step_2_without_answer.jsonl"
    with open(test_data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # 只测试前几个样本
    test_data = data[:test_cases]
    
    # 测试每个样本
    for item in test_data:
        case_id = item["id"]
        content = item["feature_content"].strip()
        
        # 构造推理prompt - 使用JSON格式
        prompt = (
            "你是一位经验丰富的医生，专门从事疾病诊断工作。请仔细阅读以下病历信息，"
            "根据患者的症状、体征和检查结果，给出准确的诊断和详细的诊断依据。\n\n"
            f"病历信息：\n{content}\n\n"
            "请以以下JSON格式输出您的诊断：\n"
            "{\n"
            '  "diseases": "简洁的疾病名称，多个疾病用逗号分隔",\n'
            '  "reason": "详细的诊断依据，包括关键症状和分析"\n'
            "}\n"
        )
        
        # 使用tokenizer处理输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        )
        
        # 移动到正确的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )
        
        # 解码结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 处理输出 - 提取生成的部分
        completion = generated_text[len(prompt):].strip()
        
        # 打印原始输出
        logger.info(f"样本ID: {case_id}")
        logger.info(f"原始输出:\n{completion}")
        
        # 尝试解析JSON
        json_result = extract_json(completion)
        if json_result and "diseases" in json_result and "reason" in json_result:
            logger.info(f"成功解析JSON格式输出:")
            logger.info(f"疾病: {json_result['diseases']}")
            logger.info(f"诊断依据: {json_result['reason'][:100]}...")
        else:
            logger.warning(f"JSON解析失败，尝试备用方法")
            
            # 尝试使用"诊断依据："分隔
            if "诊断依据：" in completion:
                parts = completion.split("诊断依据：", 1)
                disease = parts[0].strip()
                reason = parts[1].strip()
                logger.info(f"使用分隔符解析:")
                logger.info(f"疾病: {disease}")
                logger.info(f"诊断依据: {reason[:100]}...")
            else:
                # 尝试使用第一行作为疾病名称
                lines = completion.split('\n')
                if len(lines) > 1 and len(lines[0]) < 50:  # 首行较短，可能是疾病名
                    disease = lines[0].strip()
                    reason = '\n'.join(lines[1:]).strip()
                    logger.info(f"使用首行解析:")
                    logger.info(f"疾病: {disease}")
                    logger.info(f"诊断依据: {reason[:100]}...")
                else:
                    # 如果都失败，将整个输出作为reason
                    logger.warning(f"所有解析方法都失败")
        
        logger.info("-" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python test_json_format.py <model_base> <adapter_dir> [test_cases]")
        print("例如: python test_json_format.py ../models/qwen/Qwen2___5-7B-Instruct ../models/qwen_lora_finetuned/20250316_022609/final_model 3")
        sys.exit(1)
    
    model_base = sys.argv[1]
    adapter_dir = sys.argv[2]
    test_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    test_json_format(model_base, adapter_dir, test_cases) 