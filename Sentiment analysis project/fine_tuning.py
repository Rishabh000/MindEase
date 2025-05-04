# scripts/fine_tune.py

import os
import json
import torch
import pandas as pd
from collections import Counter
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from clearml import Task, Dataset as ClearMLDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def main():
    # Step 1: Initialize ClearML Task
    task = Task.init(project_name="Emotion Detection", task_name="Fine-tuning Emotion Model")

    # Step 2: Get the latest dataset from ClearML
    dataset_path = ClearMLDataset.get(
        dataset_project="Emotion Detection Datasets",  # Replace with your ClearML dataset project name if different
        dataset_name="emotion_data",  # Replace with your ClearML dataset name if different
    ).get_local_copy()

    print(f"✅ Pulled dataset from ClearML server at: {dataset_path}")

    # Step 3: Load the sampled dataset
    df = pd.read_csv(os.path.join(dataset_path, "sampled_combined_emotion_data.csv"))

    # Step 4: Load label mapping
    with open(os.path.join(dataset_path, "label_mapping.json"), "r") as f:
        label_mapping = json.load(f)

    # Step 5: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(dataset_path, "tokenizer"))

    # Step 6: Prepare HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(lambda x: {'labels': label_mapping[x['emotion']]})
    dataset = dataset.remove_columns(['text', 'emotion'])
    dataset.set_format("torch")

    # Step 7: Split into train and validation
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Quick Label Distribution Check
    labels_list = [int(label) for label in eval_dataset['labels']]
    label_counter = Counter(labels_list)
    print("\nSummary of label distribution in evaluation dataset:")
    for label_id, count in label_counter.items():
        print(f"Label {label_id}: {count} samples")

    # Step 8: Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 9: Load the previously trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        "../outputs/final_model/",  # Fine-tuning starting point
        num_labels=len(label_mapping)
    )

    # Step 10: Fine-tuning Arguments
    os.makedirs("../outputs/fine_tuned_checkpoints", exist_ok=True)
    os.makedirs("../outputs/fine_tuned_logs", exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./outputs/fine_tuned_checkpoints/",
        evaluation_strategy="steps",
        eval_steps=10,
        save_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_total_limit=2,
        logging_dir="./outputs/fine_tuned_logs/",
        logging_steps=5,
        report_to=["clearml"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        learning_rate=2e-5,  # Small learning rate for fine-tuning
    )

    # Step 11: Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 12: Fine-tune the Model
    trainer.train()

    # Step 13: Save fine-tuned model separately
    os.makedirs("../outputs/fine_tuned_model", exist_ok=True)
    model.save_pretrained("../outputs/fine_tuned_model/")
    tokenizer.save_pretrained("../outputs/fine_tuned_model/")

    print("[OK] Fine-tuning completed and model saved successfully!")

if __name__ == "__main__":
    main()
