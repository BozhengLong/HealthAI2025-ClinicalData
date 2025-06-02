#!/usr/bin/env python3
"""
HealthAI 2025 比赛 - API知识蒸馏引擎
专为HealthAI 2025临床医学数据处理竞赛设计的API蒸馏方案

比赛信息：https://competition.pkucxpl.com/
- 教师模型：Qwen2.5-72B-Instruct (通过开源API调用，符合比赛规则)
- 学生模型：Qwen2.5-7B-Instruct (比赛指定模型)
- 技术路线：API蒸馏 + 权重优化 + 结构化医疗输出
"""

import os
import sys
import json
import time
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openai import OpenAI
from transformers import AutoTokenizer
import re

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

@dataclass
class APIDistillationConfig:
    """API蒸馏配置"""
    # API配置
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    teacher_model_name: str = "qwen2.5-72b-instruct"
    
    # 推理权重配置
    reasoning_weight: float = 3.0
    diagnosis_weight: float = 2.5
    feature_weight: float = 0.3
    
    # 训练配置
    temperature: float = 0.7
    max_tokens: int = 800
    use_progressive: bool = True
    
    # 数据配置
    max_length: int = 1024
    batch_size: int = 4
    learning_rate: float = 2e-5

class MedicalReasoningExtractor:
    """医疗推理步骤提取器"""
    
    def __init__(self):
        self.reasoning_patterns = [
            r"分析[:：]?", r"考虑[:：]?", r"因为|由于",
            r"所以|因此|故", r"表明|提示|显示",
            r"排除|除外", r"鉴别|区分", r"建议|推荐"
        ]
        
        self.diagnosis_patterns = [
            r"诊断[:：]?", r"疾病[:：]?", r"病症|症状",
            r"结论[:：]?", r"确诊|初诊|临床诊断"
        ]
        
        self.feature_patterns = [
            r"患者|病人", r"主诉|现病史", r"体格检查|查体",
            r"实验室检查|辅助检查", r"既往史|过敏史"
        ]
    
    def extract_reasoning_labels(self, text: str, tokenizer) -> Dict[str, Any]:
        """提取推理标签和权重"""
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_texts = [tokenizer.decode([token]) for token in tokens]
            
            weights = torch.ones(len(tokens))
            labels = {
                'reasoning': torch.zeros(len(tokens)),
                'diagnosis': torch.zeros(len(tokens)),
                'feature': torch.zeros(len(tokens))
            }
            
            for i, token_text in enumerate(token_texts):
                token_lower = token_text.lower().strip()
                
                if any(re.search(pattern, token_lower) for pattern in self.reasoning_patterns):
                    labels['reasoning'][i] = 1.0
                    weights[i] = 3.0
                elif any(re.search(pattern, token_lower) for pattern in self.diagnosis_patterns):
                    labels['diagnosis'][i] = 1.0
                    weights[i] = 2.5
                elif any(re.search(pattern, token_lower) for pattern in self.feature_patterns):
                    labels['feature'][i] = 1.0
                    weights[i] = 0.3
            
            return {
                'tokens': tokens,
                'weights': weights,
                'reasoning_labels': labels['reasoning'],
                'diagnosis_labels': labels['diagnosis'],
                'feature_labels': labels['feature'],
                'reasoning_ratio': labels['reasoning'].mean().item(),
                'diagnosis_ratio': labels['diagnosis'].mean().item(),
                'feature_ratio': labels['feature'].mean().item()
            }
        except Exception as e:
            logger.warning(f"推理标签提取失败: {e}")
            return {
                'tokens': [],
                'weights': torch.ones(1),
                'reasoning_ratio': 0.0,
                'diagnosis_ratio': 0.0,
                'feature_ratio': 0.0
            }

class APIDistillationEngine:
    """API蒸馏引擎"""
    
    def __init__(self, config: APIDistillationConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.reasoning_extractor = MedicalReasoningExtractor()
        self.error_handler = ErrorHandler()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
    
    def create_medical_prompt(self, content: str) -> str:
        """创建医疗诊断提示"""
        return f"""作为一名资深医生，请根据患者的症状和检查结果进行详细分析诊断。

请按照以下步骤进行推理：
1. 分析患者的主要症状和体征
2. 考虑相关的检查结果和病史
3. 进行鉴别诊断，排除不太可能的疾病
4. 得出最可能的诊断结论和治疗建议

患者信息：
{content}

请以JSON格式输出诊断结果，包含：
- reason: 详细的诊断推理过程
- diseases: 最可能的疾病诊断  
- feature_content: 患者信息（原始输入）

诊断结果："""
    
    def generate_diagnosis(self, content: str) -> Dict[str, Any]:
        """生成诊断结果"""
        prompt = self.create_medical_prompt(content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.teacher_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            # 解析JSON
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                else:
                    result = {
                        "reason": response_text,
                        "diseases": "需要进一步检查",
                        "feature_content": content
                    }
            except json.JSONDecodeError:
                result = {
                    "reason": response_text,
                    "diseases": "需要进一步检查", 
                    "feature_content": content
                }
            
            return {
                'success': True,
                'result': result,
                'full_response': response_text,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'result': None
            }
    
    def generate_training_data(self, 
                             raw_data: List[Dict], 
                             output_file: str,
                             max_samples: int = None) -> List[Dict]:
        """生成训练数据"""
        training_samples = []
        failed_count = 0
        
        if max_samples:
            raw_data = raw_data[:max_samples]
        
        logger.info(f"开始生成API训练数据: {len(raw_data)} 个样本")
        
        for i, sample in enumerate(raw_data):
            content = sample.get('feature_content', '')
            if not content:
                logger.warning(f"样本 {i} 缺少feature_content，跳过")
                continue
            
            logger.info(f"处理样本 {i+1}/{len(raw_data)}")
            
            # 调用API生成诊断
            result = self.generate_diagnosis(content)
            
            if result['success']:
                diagnosis_result = result['result']
                full_response = result['full_response']
                
                training_sample = {
                    'id': sample.get('id', i),
                    'original_content': content,
                    'prompt': result['prompt'],
                    'response': full_response,
                    'full_text': result['prompt'] + "\n" + full_response,
                    'structured_result': diagnosis_result,
                    'reason': diagnosis_result.get('reason', ''),
                    'diseases': diagnosis_result.get('diseases', ''),
                    'feature_content': diagnosis_result.get('feature_content', content)
                }
                
                training_samples.append(training_sample)
                logger.info(f"样本 {i+1} 生成成功")
                
                # 每处理10个样本保存一次
                if (i + 1) % 10 == 0:
                    self._save_intermediate_data(training_samples, output_file)
                
            else:
                failed_count += 1
                logger.warning(f"样本 {i+1} 生成失败: {result.get('error', 'Unknown error')}")
            
            # API限流
            time.sleep(0.5)
        
        # 保存最终数据
        self._save_training_data(training_samples, output_file)
        
        logger.info(f"API数据生成完成: {len(training_samples)} 成功, {failed_count} 失败")
        return training_samples
    
    def _save_intermediate_data(self, data: List[Dict], output_file: str):
        """保存中间数据"""
        temp_file = output_file.replace('.json', '_temp.json')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存中间数据失败: {e}")
    
    def _save_training_data(self, data: List[Dict], output_file: str):
        """保存最终训练数据"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'num_samples': len(data),
                        'description': 'API-based distillation training data',
                        'teacher_model': self.config.teacher_model_name,
                        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'data': data
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"训练数据已保存: {output_file}")
        except Exception as e:
            self.error_handler.handle_error(e, "保存训练数据失败")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="API知识蒸馏数据生成")
    parser.add_argument("--api_key", type=str, required=True, help="API密钥")
    parser.add_argument("--input_file", type=str, required=True, help="输入数据文件")
    parser.add_argument("--output_file", type=str, required=True, help="输出训练数据文件")
    parser.add_argument("--max_samples", type=int, default=None, help="最大处理样本数")
    parser.add_argument("--teacher_model", type=str, 
                        default="qwen2.5-72b-instruct", help="Teacher模型名称")
    
    args = parser.parse_args()
    
    # 创建配置
    config = APIDistillationConfig(
        api_key=args.api_key,
        teacher_model_name=args.teacher_model
    )
    
    # 加载原始数据
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.jsonl'):
                raw_data = []
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
            else:
                raw_data = json.load(f)
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return
    
    # 创建API蒸馏引擎
    engine = APIDistillationEngine(config)
    
    # 生成训练数据
    engine.generate_training_data(
        raw_data=raw_data,
        output_file=args.output_file,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main() 