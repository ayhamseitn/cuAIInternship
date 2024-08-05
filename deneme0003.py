import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def soru_cevap_sistemi(model_name, context, question):
    # Model ve tokenizer'ı yükleme
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # Context'i parçalama
    chunk_size = 512
    chunks = []
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i + chunk_size]
        inputs = tokenizer.encode_plus(question, chunk, return_tensors="pt")
        chunks.append((inputs, chunk))

    # Modelden tahmin al
    start_scores = []
    end_scores = []
    max_len = 0
    for inputs, chunk in chunks:
        outputs = model(**inputs)
        s = outputs.start_logits
        e = outputs.end_logits
        start_scores.append(s)
        end_scores.append(e)
        max_len = max(max_len, s.shape[1])

    # Pad the tensors to the maximum length
    padded_start_scores = []
    padded_end_scores = []
    for s, e in zip(start_scores, end_scores):
        padding = torch.zeros((1, max_len - s.shape[1]))
        padded_s = torch.cat((s, padding), dim=1)
        padded_e = torch.cat((e, padding), dim=1)
        padded_start_scores.append(padded_s)
        padded_end_scores.append(padded_e)

    # Concatenate the padded tensors
    start_scores = torch.cat(padded_start_scores, dim=0)
    end_scores = torch.cat(padded_end_scores, dim=0)

    # En yüksek skorları alarak cevabı belirleme
    max_start_indices = torch.argmax(start_scores)
    max_end_indices = torch.argmax(end_scores) + 1

    # Cevabı alma
    answer_tokens = []
    for inputs, chunk in chunks:
        answer_tokens.extend(inputs["input_ids"][0][max_start_indices:max_end_indices])
    answer = tokenizer.decode(answer_tokens)
    return answer

# Load JSON data from file
with open('C:\\Users\\Dell\\Desktop\\Staj\\cuAI01\\data1sonSpan.json', encoding='utf-8') as f:
    data = json.load(f)

# Extract story and questions from JSON dataa
story = data['data']['story']
questions = data['data']['questions']
answers = data['data']['answers']

# Process each question
for question, answer in zip(questions, answers):
    input_text = question['input_text']
    cevap = soru_cevap_sistemi("salti/bert-base-multilingual-cased-finetuned-squad", story, input_text)
    print("Question:", input_text)
    print("Answer:", cevap)
    print()