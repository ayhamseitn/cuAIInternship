import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Model ve tokenizer yükleme
model_name = "salti/bert-base-multilingual-cased-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# JSON dosyasını okuma
with open("C:/Users/Dell/Desktop/Staj/cuAI01/test00.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def find_best_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def split_into_chunks(text, max_length):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
    return chunks

# Kullanıcıdan soru alma
user_question = input("Sorunuzu girin: ")

# En iyi cevabı bulma
best_answer = ""
best_score = float("-inf")
for section in data:
    for paragraph in section["paragraphs"]:
        context = paragraph["context"]
        # Token uzunluğunu hesapla ve parçalara ayır
        max_length = 512 - len(tokenizer.tokenize(user_question)) - 3
        chunks = split_into_chunks(context, max_length)

        for chunk in chunks:
            answer = find_best_answer(user_question, chunk)
            inputs = tokenizer.encode_plus(user_question, chunk, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            score = torch.max(outputs.start_logits) + torch.max(outputs.end_logits)

            if score > best_score:
                best_score = score
                best_answer = answer

print(f"En iyi cevap: {best_answer}")
