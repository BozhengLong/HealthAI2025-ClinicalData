#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用微调后的模型进行推理，使用JSON格式的提示和输出，
并实现增量保存、内存管理和错误处理功能。
"""

import json
import argparse
import os
import logging
import gc
import re
import time
import traceback
from tqdm import tqdm
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("infer_json.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_memory_usage():
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logger.info(f"GPU {i}: 已分配 {memory_allocated:.2f} GB, 已保留 {memory_reserved:.2f} GB")

def save_results(results, output_file):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"结果已保存到 {output_file}")

def load_previous_results(output_file):
    """加载之前的结果"""
    results = []
    processed_ids = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    results.append(result)
                    processed_ids.add(result['id'])
                except:
                    logger.warning(f"无法解析结果行: {line}")
    
    logger.info(f"已加载 {len(results)} 条之前的结果")
    return results, processed_ids

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
                logger.warning(f"JSON解析失败: {json_str[:100]}...")
                return None
    return None

def parse_output(completion, item_id):
    """解析模型输出，尝试多种方法提取疾病和诊断依据"""
    result = {
        "id": item_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 尝试解析JSON
    json_result = extract_json(completion)
    if json_result and "diseases" in json_result and "reason" in json_result:
        result["diseases"] = json_result["diseases"]
        result["reason"] = json_result["reason"]
        result["parse_method"] = "json"
        return result
    
    # 尝试使用"诊断依据："分隔
    if "诊断依据：" in completion:
        parts = completion.split("诊断依据：", 1)
        disease = parts[0].strip()
        reason = parts[1].strip()
        result["diseases"] = disease
        result["reason"] = reason
        result["parse_method"] = "separator"
        return result
    
    # 尝试使用第一行作为疾病名称
    lines = completion.split('\n')
    if len(lines) > 1 and len(lines[0]) < 50:  # 首行较短，可能是疾病名
        disease = lines[0].strip()
        reason = '\n'.join(lines[1:]).strip()
        result["diseases"] = disease
        result["reason"] = reason
        result["parse_method"] = "first_line"
        return result
    
    # 如果都失败，将整个输出作为reason
    result["diseases"] = "解析失败"
    result["reason"] = completion
    result["parse_method"] = "failed"
    return result

def main():
    parser = argparse.ArgumentParser(description="使用微调后的模型进行推理")
    parser.add_argument("--model_base", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter_dir", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--data_path", type=str, default="../data/camp_data_step_2_without_answer.jsonl", help="测试数据路径")
    parser.add_argument("--output_file", type=str, default="results_json.jsonl", help="输出文件路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成的最大token数")
    parser.add_argument("--save_interval", type=int, default=5, help="保存结果的间隔（样本数）")
    parser.add_argument("--start_index", type=int, default=0, help="开始处理的样本索引")
    parser.add_argument("--end_index", type=int, default=None, help="结束处理的样本索引")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.8, help="top-p采样参数")
    parser.add_argument("--top_k", type=int, default=20, help="top-k采样参数")
    
    args = parser.parse_args()
    
    # 加载之前的结果
    results, processed_ids = load_previous_results(args.output_file)
    
    # 加载测试数据
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # 确定处理范围
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(data)
    data_to_process = data[start_idx:end_idx]
    
    # 过滤掉已处理的样本
    data_to_process = [item for item in data_to_process if item["id"] not in processed_ids]
    
    if not data_to_process:
        logger.info("没有新的样本需要处理")
        return
    
    logger.info(f"将处理 {len(data_to_process)} 个样本")
    
    try:
        # 加载模型和分词器
        logger.info(f"加载基础模型: {args.model_base}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_base,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info(f"加载 LoRA 适配器: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_dir,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 打印初始内存使用情况
        print_memory_usage()
        
        # 处理每个样本
        for i, item in enumerate(tqdm(data_to_process)):
            item_id = item["id"]
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
            
            try:
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
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
                
                # 解码结果
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 处理输出 - 提取生成的部分
                completion = generated_text[len(prompt):].strip()
                
                # 解析输出
                result = parse_output(completion, item_id)
                
                # 添加原始输出以供参考
                result["raw_output"] = completion
                
                # 添加到结果列表
                results.append(result)
                
                # 定期保存结果
                if (i + 1) % args.save_interval == 0:
                    save_results(results, args.output_file)
                    logger.info(f"已处理 {i + 1}/{len(data_to_process)} 个样本")
                    print_memory_usage()
                
                # 清理内存
                del inputs, outputs, generated_text
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"处理样本 {item_id} 时出错: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 记录错误信息
                error_result = {
                    "id": item_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "diseases": "处理错误",
                    "reason": f"错误: {str(e)}",
                    "parse_method": "error",
                    "error": str(e)
                }
                results.append(error_result)
                
                # 尝试清理内存
                torch.cuda.empty_cache()
                gc.collect()
        
        # 最终保存结果
        save_results(results, args.output_file)
        logger.info("推理完成")
        print_memory_usage()
        
    except Exception as e:
        logger.error(f"运行时错误: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 保存已处理的结果
        if results:
            save_results(results, args.output_file)
            logger.info(f"已保存 {len(results)} 条结果")
    
    finally:
        # 清理内存
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"总运行时间: {elapsed_time:.2f} 秒") 