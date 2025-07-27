import json
import re

def analyze_jsonl_sentences(file_path):
    questions = []
    all_sentences = []
    
    # 1. 读取JSONL文件并提取question
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                questions.append(record["question"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析错误: {e}，跳过该行")
    
    # 2. 分割句子并过滤最后问句
    for q in questions:
        sentences = re.split(r'\.\s+', q)  # 按句号分割
        if sentences and len(sentences) > 1:  # 确保有可分割句子
            # 移除空字符串并排除最后一句
            valid_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
            all_sentences.extend(valid_sentences)
    
    # 3. 计算平均长度
    if not all_sentences:
        return 0  # 避免除零错误
    
    total_chars = sum(len(s) for s in all_sentences)
    avg_length = total_chars / len(all_sentences)
    
    # 返回结果（平均长度 + 总句子数）
    return avg_length, len(all_sentences)

# 调用示例
avg_len, sentence_count = analyze_jsonl_sentences("work/Crowdsourcing/MLDR/MLDR/datasets/Self_define/full.jsonl")
print(f"分析完成！共处理 {sentence_count} 个句子，平均长度: {avg_len:.2f} 字符")