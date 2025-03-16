#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估模型输出的JSON格式解析成功率
"""

import json
import argparse
import os
import logging
import re
from collections import Counter

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
            return result, "json_success"
        except json.JSONDecodeError:
            # 如果解析失败，尝试修复常见的JSON格式问题
            try:
                # 替换多余的引号和转义字符
                fixed_json = re.sub(r'\\+', r'\\', json_str)
                # 尝试再次解析
                result = json.loads(fixed_json)
                return result, "json_fixed"
            except:
                return None, "json_failed"
    return None, "no_json"

def evaluate_results(results_file):
    """评估结果文件中的JSON解析成功率"""
    if not os.path.exists(results_file):
        logger.error(f"结果文件不存在: {results_file}")
        return
    
    # 统计信息
    total = 0
    parse_methods = Counter()
    json_success = 0
    has_diseases = 0
    has_reason = 0
    
    # 读取结果文件
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                total += 1
                
                # 统计解析方法
                if 'parse_method' in result:
                    parse_methods[result['parse_method']] += 1
                
                # 检查原始输出
                if 'raw_output' in result:
                    raw_output = result['raw_output']
                    json_obj, status = extract_json(raw_output)
                    
                    if status in ['json_success', 'json_fixed']:
                        json_success += 1
                        
                        # 检查JSON字段
                        if json_obj and 'diseases' in json_obj:
                            has_diseases += 1
                        if json_obj and 'reason' in json_obj:
                            has_reason += 1
            except:
                logger.warning(f"无法解析结果行")
    
    # 计算统计数据
    if total > 0:
        json_success_rate = json_success / total * 100
        diseases_rate = has_diseases / total * 100
        reason_rate = has_reason / total * 100
        complete_json_rate = (has_diseases and has_reason) / total * 100 if total > 0 else 0
        
        # 打印统计信息
        logger.info(f"总样本数: {total}")
        logger.info(f"JSON解析成功率: {json_success_rate:.2f}%")
        logger.info(f"包含diseases字段率: {diseases_rate:.2f}%")
        logger.info(f"包含reason字段率: {reason_rate:.2f}%")
        logger.info(f"完整JSON率 (包含diseases和reason): {complete_json_rate:.2f}%")
        logger.info(f"解析方法统计: {dict(parse_methods)}")
    else:
        logger.warning("没有找到有效的结果")

def main():
    parser = argparse.ArgumentParser(description="评估模型输出的JSON格式解析成功率")
    parser.add_argument("--results_file", type=str, required=True, help="结果文件路径")
    
    args = parser.parse_args()
    evaluate_results(args.results_file)

if __name__ == "__main__":
    main() 