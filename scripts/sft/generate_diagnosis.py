## Author: Bozheng Long
## Date Created: 2025-03-15
## Last Modified: 2025-03-16
## Description: 使用 Qwen_2.5-72B-instruct 模型生成诊断结果

import os
import json
import time
import requests
import logging
import sys
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# 导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_project_root, get_data_path, setup_logging, ensure_dir

# 设置日志
log_file = Path(get_project_root()) / "logs" / f"label_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logging(log_file=log_file)

# 配置 API Key 和 API 基础 URL - 使用环境变量
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("请设置OPENAI_API_KEY环境变量")

api_base = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1/chat/completions")

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

logger.info(f"使用API基础URL: {api_base}")

# 配置文件路径
INPUT_PATH = get_data_path("camp_data_step_1_without_answer.jsonl")    # 输入数据，每条记录包含 id 和 feature_content
OUTPUT_PATH = get_data_path("train_labeled_by_deepseek.jsonl")    # 输出文件，将生成训练数据
CHECKPOINT_PATH = get_data_path("checkpoint.jsonl")  # 用于保存中间结果

def parse_output(reply):
    try:
        result = json.loads(reply)
        diseases = result.get("diseases", "").strip()
        reason = result.get("reason", "").strip()
        return diseases, reason
    except Exception:
        return "", reply

def main():
    # 读取病例数据
    logger.info(f"从 {INPUT_PATH} 读取数据")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]  # 测试时用前10条，确认无误后删除 [:10]

    # 如果有检查点文件，则加载已经处理的记录的ID
    processed_ids = set()
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"发现检查点文件: {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_ids.add(rec["id"])
                except Exception as e:
                    logger.warning(f"解析检查点记录时出错: {e}")
                    continue

    output_data = []

    # 从已处理的记录之后继续处理
    remaining_data = [d for d in data if d["id"] not in processed_ids]

    logger.info(f"开始处理，总记录数：{len(data)}, 已处理：{len(processed_ids)}, 未处理：{len(remaining_data)}")

checkpoint_interval = 50  # 每50条保存一次

for idx, item in enumerate(tqdm(remaining_data)):
    case_id = item["id"]
    content = item["feature_content"].strip()

        prompt = f"""你是一位经验丰富的医生，请根据以下病历信息进行诊断。
请严格按照以下 JSON 格式输出，不要输出其他内容：
{{
  "diseases": "<病名>",
  "reason": "<详细诊断依据>"
}}
其中，"diseases" 字段只应包含简短的疾病名称（如"贫血"、"糖尿病"等），
而 "reason" 字段应包含详细的诊断依据。
病历信息如下：
{content}"""

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]

    reply = ""
    max_retries = 3
    success = False
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                    # model="qwen2.5-72b-instruct", 
                    model="Pro/deepseek-ai/DeepSeek-R1",
                messages=messages,
                stream=False,
            )
            reply = completion.choices[0].message.content
            success = True
            break
        except Exception as e:
                logger.error(f"[Error @ {case_id}]: {e} (attempt {attempt+1}/{max_retries})")
            time.sleep(2 ** attempt)  # 指数退避

    if not success:
            logger.error(f"[Failure @ {case_id}]: 放弃处理")
        output_data.append({
            "id": case_id,
            "diseases": "",
            "reason": "",
            "feature_content": content
        })
    else:
        diseases, reason = parse_output(reply)
        output_data.append({
            "id": case_id,
            "diseases": diseases,
            "reason": reason,
            "feature_content": content
        })

    # 保存检查点
    if (idx + 1) % checkpoint_interval == 0:
        with open(CHECKPOINT_PATH, "a", encoding="utf-8") as cp:
            for rec in output_data:
                cp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        output_data = []  # 清空当前保存的部分
            logger.info(f"已保存 {idx+1} 条记录的检查点")
    
    time.sleep(1.2)

# 保存剩余的记录
if output_data:
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as cp:
        for rec in output_data:
            cp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
# 最终将 CHECKPOINT 文件复制为 OUTPUT 文件
os.rename(CHECKPOINT_PATH, OUTPUT_PATH)
    logger.info(f"标签生成完成，输出文件保存在：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()