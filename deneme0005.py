import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Model ve tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")

def split_text(text, max_length=512):
    """Uzun metni belirtilen uzunlukta parçalara ayırır."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(tokenizer.encode(word, add_special_tokens=False))
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def question_answer(question, story):
    # Story'yi parçalara ayır
    paragraphs = split_text(story, max_length=512)
    best_answer = ""
    best_score = float('-inf')

    for paragraph in paragraphs:
        inputs = tokenizer.encode_plus(question, paragraph, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        # En yüksek skora sahip cevabı seç
        score = torch.max(answer_start_scores) + torch.max(answer_end_scores)
        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer

def main():
    # Kullanıcıdan JSON veri dosyasının yolunu al
    file_path = input("Lütfen JSON dosyasının yolunu girin: ")

    # JSON dosyasını oku
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Story'yi ve soruları al
    story = data['data']['story']
    questions = data['data']['questions']

    # Soruları listele
    for idx, item in enumerate(questions):
        question = item.get('input_text')
        if question:
            print(f"{idx}: {question}")
        else:
            print(f"{idx}: Sorunun anahtarı bulunamadı.")

    # Kullanıcıdan bir soru seçmesini iste
    question_index = int(input("Sormak istediğiniz sorunun indeksini seçin: "))
    selected_question = questions[question_index].get('input_text', 'Soru bulunamadı.')

    # Soruyu yanıtla
    answer = question_answer(selected_question, story)

    print(f"Soru: {selected_question}")
    print(f"Cevap: {answer}")

if __name__ == "__main__":
    main()
