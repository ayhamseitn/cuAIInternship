import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Model ve tokenizer'ı yükle
tokenizer = BertTokenizer.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("salti/bert-base-multilingual-cased-finetuned-squad")

def question_answer(question, story):
    # Story'yi parçalara böl
    paragraphs = story.split("\n\n")
    best_answer = ""
    best_score = float('-inf')

    for paragraph in paragraphs:
        inputs = tokenizer.encode_plus(question, paragraph, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        if len(input_ids) > 512:
            print("Skipping a paragraph as it exceeds the maximum length")
            continue

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

    print(f"Soru: {question}")
    print(f"Cevap: {best_answer}")

def main():
    # Kullanıcıdan CoQA veri dosyasının yolunu al
    file_path = input("Lütfen CoQA veri dosyanızın yolunu girin: ")

    # JSON dosyasını oku
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Story'yi ve soruları al
    story = data['data']['story']
    questions = [
        "Farklı yükseköğretim kurumlarının diploma programları arasında yatay geçiş yapılabilir mi?",
        "Önlisans ve lisans diploma programlarının hangi dönemlerine yatay geçiş yapılamaz?",
        "Hangi durumlarda yatay geçiş yapılamaz?",
        "Kurum (üniversite) içi programlar arası yatay geçişlerde hangi şart aranır?",
        "Yatay geçiş başvurusu yapacak öğrencilerin disiplin cezası almamış olmaları gerekir mi?",
        "Yatay geçiş değerlendirme sonuçları nerede duyurulur?",
        "Yatay geçiş hakkı kazanan adaylar, kararlarını nasıl öğrenirler?",
        "Yatay geçiş başvuruları ne zaman yapılır?",
        "Yatay geçiş başvuruları nasıl değerlendirilir?",
        "Hangi durumlarda yatay geçiş başvuruları işleme konulmaz?",
        "Yatay geçiş yapan öğrencilerin önceki yükseköğretim kurumunda aldıkları dersler nasıl değerlendirilir?",
        "Eşdeğer dersler belirlenirken hangi kriter dikkate alınır?",
        "Yatay geçiş yapan öğrenciler eşdeğer derslerden muaf tutulur mu?",
        "Yatay geçiş yapan öğrencilerin yeni programa intibakları kim tarafından yapılır?",
        "Önceki diploma programında aldığım dersler transkriptime işlenir mi?",
        "İntibak işlemleri tamamlandıktan sonra ne olur?",
        "İntibak edilen derslerin notları nasıl kabul edilir?",
        "Ders intibak işlemleri sırasında hangi kriter dikkate alınır?",
        "Bu yönerge hükümleri kim tarafından yürütülür?"
    ]

    # Soruları listele
    for idx, question in enumerate(questions):
        print(f"{idx}: {question}")

    # Kullanıcıdan bir soru seçmesini iste
    question_index = int(input("Sormak istediğiniz sorunun indeksini seçin: "))
    selected_question = questions[question_index]

    # Soruyu yanıtla
    question_answer(selected_question, story)

if __name__ == "__main__":
    main()
