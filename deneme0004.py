import json
import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Model and tokenizer initialization
model = BertForQuestionAnswering.from_pretrained('salti/bert-base-multilingual-cased-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('salti/bert-base-multilingual-cased-finetuned-squad')

def question_answer(question, text):
    # Tokenize question and text as a pair
    inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    # Model output using input_ids and token_type_ids
    outputs = model(input_ids, token_type_ids=token_type_ids)
    
    # Reconstructing the answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

    if answer.startswith("[CLS]") or answer == "":
        answer = "Unable to find the answer to your question."
    
    print("\nPredicted answer:\n{}".format(answer.capitalize()))

def load_coqa_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    file_path = input("Please enter the path to your CoQA data file: \n")
    coqa_data = load_coqa_data(file_path)
    
    # Display available texts for selection
    story = coqa_data['data']['story']
    print(f"Story: {story[:100]}...")  # Displaying only first 100 characters of the story for brevity
    
    questions = coqa_data['data']['questions']
    for i, question in enumerate(questions):
        print(f"{i}: {question['input_text']}")
    
    while True:
        selected_index = int(input("\nSelect the index of the question you want to ask: "))
        question = questions[selected_index]['input_text']
        
        question_answer(question, story)
        
        flag = True
        flag_N = False
        
        while flag:
            response = input("\nDo you want to ask another question based on this story (Y/N)? ")
            if response[0].upper() == "Y":
                flag = False
            elif response[0].upper() == "N":
                print("\nBye!")
                flag = False
                flag_N = True
                
        if flag_N:
            break

if __name__ == "__main__":
    main()
