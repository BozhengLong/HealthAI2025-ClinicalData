import json
import argparse
import os
import logging
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_results(output_data, output_path, is_final=False):
    """保存结果到输出文件"""
    save_path = output_path if is_final else f"{output_path}.partial"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            for rec in output_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"已保存 {len(output_data)} 条结果到: {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        return False

def load_previous_results(output_path):
    """加载之前的处理结果（用于断点续传）"""
    output_data = []
    processed_ids = set()
    
    # 先尝试加载正式文件
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    output_data.append(item)
                    processed_ids.add(item["id"])
            logger.info(f"从{output_path}加载了 {len(output_data)} 条已处理的结果")
        except Exception as e:
            logger.warning(f"加载{output_path}失败: {str(e)}")
    
    # 如果正式文件没有或为空，尝试加载部分结果文件
    if len(output_data) == 0 and os.path.exists(f"{output_path}.partial"):
        try:
            with open(f"{output_path}.partial", "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    output_data.append(item)
                    processed_ids.add(item["id"])
            logger.info(f"从{output_path}.partial加载了 {len(output_data)} 条已处理的结果")
        except Exception as e:
            logger.warning(f"加载{output_path}.partial失败: {str(e)}")
    
    return output_data, processed_ids

def print_memory_usage():
    """打印当前内存使用情况"""
    if torch.cuda.is_available():
        alloc_mem = torch.cuda.memory_allocated() / (1024**3)  # 转换为GB
        max_mem = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"GPU内存：当前 {alloc_mem:.2f} GB，峰值 {max_mem:.2f} GB")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default="../models/qwen/Qwen2___5-7B-Instruct",
                        help="原始 Qwen 基础模型名称或路径")
    parser.add_argument("--adapter_dir", type=str, default="../models/qwen_lora_finetuned/20250316_044623/final_model",
                        help="LoRA 适配器所在目录，即微调后的 final_model 文件夹")
    parser.add_argument("--input_path", type=str, default="../data/camp_data_step_2_without_answer.jsonl",
                        help="测试数据文件路径")
    parser.add_argument("--output_path", type=str, default="../data/predict_output.jsonl",
                        help="生成的预测结果保存路径")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="生成新 token 的最大数目")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度，越高越随机")
    parser.add_argument("--do_sample", type=bool, default=True,
                        help="是否使用采样生成")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="每处理多少个样本保存一次结果")
    args = parser.parse_args()

    try:
        # 1. 加载基础模型
        logger.info(f"加载基础模型: {args.model_base}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_base,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 2. 加载 LoRA 适配器
        logger.info(f"加载 LoRA 适配器: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        
        # 3. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_dir,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 设置生成配置
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else 1.0,
            top_p=0.8 if args.do_sample else 1.0,
            top_k=20 if args.do_sample else 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 4. 加载测试数据
        logger.info(f"读取测试数据: {args.input_path}")
        with open(args.input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        logger.info(f"共读取 {len(data)} 条测试数据")

        # 5. 断点续传：检查是否有之前的结果
        output_data, processed_ids = load_previous_results(args.output_path)
        
        # 过滤出未处理的数据
        remaining_data = [item for item in data if item["id"] not in processed_ids]
        logger.info(f"已处理 {len(processed_ids)} 条数据，剩余 {len(remaining_data)} 条待处理")
        
        # 记录失败的样本
        failed_samples = []
        
        # 显示初始内存状态
        print_memory_usage()

        # 6. 开始逐条处理剩余数据
        logger.info("开始生成诊断结果...")
        for i, item in enumerate(tqdm(remaining_data, desc="推理进度", ncols=100)):
            try:
                case_id = item["id"]
                content = item["feature_content"].strip()
                
                # 构造推理prompt - 使用JSON格式
                prompt = (
                    "你是一位经验丰富的医生，专门从事疾病诊断工作。请仔细阅读以下病历信息，"
                    "根据患者的症状、体征和检查结果，给出准确的诊断和详细的诊断依据。\n\n"
                    f"病历信息：\n{content}n\n"
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
                    generation_config=generation_config
                )
                
                # 解码结果
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 处理输出 - 提取生成的部分
                completion = generated_text[len(prompt):].strip()
                
                # 解析结果 - 尝试提取JSON
                disease = ""
                reason = ""
                
                # 尝试解析JSON
                json_result = extract_json(completion)
                if json_result and "diseases" in json_result and "reason" in json_result:
                    disease = json_result["diseases"].strip()
                    reason = json_result["reason"].strip()
                    logger.info(f"成功解析JSON格式输出: ID={case_id}")
                else:
                    # 回退到传统解析方法
                    logger.warning(f"JSON解析失败，使用备用方法: ID={case_id}")
                    
                    # 尝试使用"诊断依据："分隔
                    if "诊断依据：" in completion:
                        parts = completion.split("诊断依据：", 1)
                        disease = parts[0].strip()
                        reason = parts[1].strip()
                    else:
                        # 尝试使用第一行作为疾病名称
                        lines = completion.split('\n')
                        if len(lines) > 1 and len(lines[0]) < 50:  # 首行较短，可能是疾病名
                            disease = lines[0].strip()
                            reason = '\n'.join(lines[1:]).strip()
                        else:
                            # 如果都失败，将整个输出作为reason
                            disease = ""
                            reason = completion
                
                # 添加到结果中
                output_data.append({
                    "id": case_id,
                    "reason": reason,
                    "diseases": disease,
                    "feature_content": content
                })
                
                # 定期增量保存结果（每处理指定数量的样本）
                if (i + 1) % args.save_interval == 0:
                    logger.info(f"已处理 {i+1}/{len(remaining_data)} 个样本，正在保存中间结果...")
                    save_results(output_data, args.output_path)
                    
                    # 定期清理内存
                    if (i + 1) % 500 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        print_memory_usage()
                
            except Exception as e:
                logger.error(f"处理样本ID {item.get('id', 'unknown')} 时出错: {str(e)}")
                failed_samples.append(item.get('id', 'unknown'))
                continue

        # 7. 保存最终结果
        logger.info("推理完成，保存最终结果...")
        save_results(output_data, args.output_path, is_final=True)
        
        # 如果有部分结果文件，完成后删除
        if os.path.exists(f"{args.output_path}.partial"):
            os.remove(f"{args.output_path}.partial")
        
        # 8. 输出统计信息
        logger.info(f"推理完成！共成功处理 {len(output_data)} 条数据")
        if failed_samples:
            logger.warning(f"处理失败的样本ID: {', '.join(failed_samples[:10])}{'...' if len(failed_samples) > 10 else ''}")
            logger.warning(f"共有 {len(failed_samples)} 个样本处理失败")
        
        print_memory_usage()
        logger.info(f"预测结果保存在: {args.output_path}")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()