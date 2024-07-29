from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

# Load the dataset
dataset = load_dataset('your_dataset_path')

# Load the model and tokenizer
model_name = "salti/bert-base-multilingual-cased-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define the preprocessing function
def preprocess_function(examples):
    questions = [q.lstrip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        sequence_ids = inputs.sequence_ids(i)

        # Find the positions of the first segment
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # Find the positions
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(context_start)
            end_positions.append(context_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define the training arguments
output_dir = "C:\\Users\\Dell\\Desktop\\Staj\\cuAI01\\Model002\\results"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)