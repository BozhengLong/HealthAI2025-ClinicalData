from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import tqdm

# 模型名称
model_name = "Qwen2.5-7B-Instruct"

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 读取数据
with open('camp_data_step_1_without_answer.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file.readlines()]

# 处理数据并生成结果
with open('submit_example.jsonl', 'w', encoding='utf-8') as output_file:
    for record in tqdm.tqdm(data):
        feature_content = record['feature_content']
        
        # 构建消息列表
        message1 = [
            {"role": "system", "content": "你是一个医生，根据给出的病例信息做出对应的诊断。"},
            {"role": "user", "content": f'病例的内容是```{feature_content}```\n\n该患者的诊断是：'}
        ]
        message2 = [
            {"role": "system", "content": "你是一个医生，请根据给出的诊断做出诊断依据。"},
            {"role": "user", "content": f'病例的内容是```{feature_content}```\n\n你的诊断依据是：'}
        ]
        
        # 应用聊天模板
        texts = [
            tokenizer.apply_chat_template(
                message1,
                tokenize=False,
                add_generation_prompt=True,
            ),
            tokenizer.apply_chat_template(
                message2,
                tokenize=False,
                add_generation_prompt=True,
            )
        ]

        # 准备模型输入
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side='left').to(model.device)

        # 生成响应
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )

        # 解码生成的文本
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        diseases, reason = responses

        # 写入结果
        result = {'id': record['id'], 'diseases': diseases, 'reason': reason}
        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')