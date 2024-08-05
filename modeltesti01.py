from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer
import torch

# Model ve tokenizer'ı yükleme
model = BigBirdForQuestionAnswering.from_pretrained('C:\\Users\\Dell\\Desktop\\Staj\\cuAI01\\Model002\\results')
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

def tokenize_qa_pair(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    return inputs

def get_answer(question, context):
    inputs = tokenize_qa_pair(question, context)
    
    # Modeli test etme
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Sonuçları çıkarma
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Token ID'lerini çıkartma
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Token'ları ve cevabı dönüştürme
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index + 1])
    return answer

# Kullanıcıdan giriş alma
def main():
    print("BigBird Soru-Cevap Modeli")
    print("Çıkmak için 'exit' yazın.")
    
    while True:
        question = input("Soru: ")
        if question.lower() == 'exit':
            break
        context = input("Bağlam (context): ")
        
        answer = get_answer(question, context)
        print(f"Cevap: {answer}")

if __name__ == "__main__":
    main()
