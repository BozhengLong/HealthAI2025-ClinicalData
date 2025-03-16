import json, time, requests
from tqdm import tqdm

# 配置
INPUT_PATH = "../data/camp_data_step_2_without_answer.jsonl"
OUTPUT_PATH = "../data/train_labeled_by_deepseek.jsonl"
API_KEY = "sk-03ebc6bf01e648679dc099427ce6c344"  # <<< 在这里填上你的 key！

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 读取病例
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f][:5]

output_data = []

def parse_output(reply):
    # 尝试直接解析 JSON
    try:
        data = json.loads(reply)
        diseases = data.get("diseases", "").strip()
        reason = data.get("reason", "").strip()
        return diseases, reason
    except Exception:
        pass
    # 如果 JSON 解析失败，则采用备用解析（如正则）
    import re
    disease_match = re.search(r"【病名】[:：]\s*(.+)", reply)
    reason_match = re.search(r"【诊断依据】[:：]\s*(.+)", reply)
    
    diseases = disease_match.group(1).strip() if disease_match else ""
    reason = reason_match.group(1).strip() if reason_match else ""
    return diseases, reason




for item in tqdm(data):
    case_id = item["id"]
    content = item["feature_content"].strip()

    # ===== 新的 Prompt 设计 =====
    # 要求模型严格输出两部分：
    # 1. 疾病名称：只给出简短病名（例如“贫血”）
    # 2. 诊断依据：详细解释、提取关键信息，不要包含多余编号或描述。

    prompt = (
        "你是一位经验丰富的医生，请根据以下病历信息进行诊断。\n"
        "请严格按照以下 JSON 格式输出，不要输出其他内容：\n"
        "{\n  \"diseases\": \"<病名>\",\n  \"reason\": \"<详细诊断依据>\"\n}\n"
        "其中，\"diseases\" 字段只应包含简短的疾病名称（如“贫血”、“糖尿病”等），而 \"reason\" 字段应包含详细的诊断依据。\n"
        "病历信息如下：\n" + content
    )


    payload = {
        "model": "deepseek-chat",  # 如需更改成你所用的模型名称，请修改此处
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    reply = ""
    max_retries = 3
    success = False
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            reply = response.json()["choices"][0]["message"]["content"]
            success = True
            break
        except requests.exceptions.ReadTimeout as rt:
            print(f"[Error @ {case_id}]: Read timed out, attempt {attempt+1} of {max_retries}")
            time.sleep(1.2)
        except Exception as e:
            print(f"[Error @ {case_id}]:", e)
            break

    if not success:
        # 如果重试后依然失败，记录空字符串，或可自行调整为其它逻辑
        output_data.append({
            "id": case_id,
            "diseases": "",
            "reason": "",
            "feature_content": content
        })
        continue

    diseases, reason = parse_output(reply)
    output_data.append({
        "id": case_id,
        "diseases": diseases,
        "reason": reason,
        "feature_content": content
    })



    # ===== 新的解析逻辑 =====
    # lines = [l.strip() for l in reply.strip().split("\n") if l.strip()]
    # disease = ""
    # reason_lines = []

    # for line in lines:
    #     if any(keyword in line for keyword in ["疾病名称", "诊断结果", "诊断为"]):
    #         parts = line.split("：")
    #         if len(parts) >= 2:
    #             disease = parts[-1].strip()
    #         continue
    #     if any(keyword in line for keyword in ["诊断依据", "理由"]):
    #         parts = line.split("：")
    #         if len(parts) >= 2:
    #             reason_lines.append(parts[-1].strip())
    #         continue
    #     reason_lines.append(line)
    
    # if not disease and lines:
    #     if len(lines[0]) < 20:
    #         disease = lines[0]
    #         reason_lines = lines[1:]
    #     else:
    #         disease = "未知"

    # reason = " ".join(reason_lines).strip()

    # ===== 输出时调整字段顺序 =====
    # 新顺序：id, reason, diseases, feature_content
    # output_data.append({
    #     "id": case_id,
    #     "diseases": diseases,
    #     "reason": reason,     
    #     "feature_content": content
    # })
    
    time.sleep(1.2)  # 避免请求过快

# ===== 写出结果 =====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for line in output_data:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")