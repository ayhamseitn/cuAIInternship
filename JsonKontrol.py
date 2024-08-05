import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def question_answer(question, story, tokenizer, model):
    # Truncate story to fit within the 512 token limit
    inputs = tokenizer.encode_plus(question, story, add_special_tokens=True, max_length=512, truncation=True, return_tensors='pt')
    
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    
    outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")

def main():
    # Load the model and tokenizer
    model_name = 'salti/bert-base-multilingual-cased-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    # Load the data file
    data_file = input("Please enter the path to your CoQA data file: ")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Inspect the structure of the JSON file
    print(json.dumps(data, indent=4, ensure_ascii=False))
    
    # Ensure the data structure matches the expected format
    if 'data' not in data or not isinstance(data['data'], list) or len(data['data']) == 0:
        print("Unexpected JSON structure.")
        return
    
    # Extract the story and questions
    story_data = data['data'][0]
    if 'story' not in story_data or 'questions' not in story_data:
        print("Missing 'story' or 'questions' in the JSON data.")
        return
    
    story = story_data['story']
    questions = [q['input_text'] for q in story_data['questions']]
    
    print(f"Story: {story[:100]}...")  # Display the first 100 characters of the story for brevity
    for idx, question in enumerate(questions):
        print(f"{idx}: {question}")
    
    # Get the user's question choice
    question_index = int(input("Select the index of the question you want to ask: "))
    question = questions[question_index]
    
    # Get the answer
    question_answer(question, story, tokenizer, model)

if __name__ == "__main__":
    main()
