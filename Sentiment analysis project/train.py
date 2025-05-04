# scripts/train.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from clearml import Task
import json
import numpy as np
import os
from collections import Counter

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def main():
    # Step 1: Initialize ClearML Task
    task = Task.init(project_name="Emotion Detection", task_name="Broad Emotion Model Training")

    # Step 2: Load dataset
    df = pd.read_csv("./data/sampled_combined_emotion_data.csv")
    dataset = Dataset.from_pandas(df)

    # Step 3: Clean dataset (VERY IMPORTANT!!)
    columns_to_remove = []
    if "__index_level_0__" in dataset.column_names:
        columns_to_remove.append("__index_level_0__")
    if "label" in dataset.column_names:
        columns_to_remove.append("label")

    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
        print(f"[INFO] Removed columns: {columns_to_remove}")

    # Step 4: Load label mapping
    with open("./data/label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    # Step 5: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./data/tokenizer/")

    # Step 6: Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)

    # Step 7: Map text labels to integer labels
    dataset = dataset.map(lambda x: {'labels': label_mapping[x['emotion']]})
    dataset = dataset.remove_columns(['text', 'emotion'])
    dataset.set_format("torch")

    # Step 8: Split into train and validation sets
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Step 9: Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 10: Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_mapping)
    )

    # Step 11: Training arguments
    os.makedirs("./outputs/checkpoints", exist_ok=True)
    os.makedirs("./outputs/logs", exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./outputs/checkpoints/",
        evaluation_strategy="steps",
        eval_steps=10,
        save_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_total_limit=2,
        logging_dir="./outputs/logs/",
        logging_steps=5,
        report_to=["clearml"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # Step 12: Print dataset verification
    print("\n Verification Before Training")
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(eval_dataset))

    labels_list_eval = [int(label) for label in eval_dataset['labels']]
    labels_list_train = [int(label) for label in train_dataset['labels']]

    # Evaluation dataset label distribution
    label_counter_eval = Counter(labels_list_eval)
    print("\n Validation dataset label distribution:")
    for label_id, count in label_counter_eval.items():
        print(f"Label {label_id}: {count} samples")

    # Training dataset label distribution
    label_counter_train = Counter(labels_list_train)
    print("\n Training dataset label distribution:")
    for label_id, count in label_counter_train.items():
        print(f"Label {label_id}: {count} samples")

    # Step 13: Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 14: Train
    trainer.train()

    # Step 15: Save final model locally
    os.makedirs("./outputs/final_model", exist_ok=True)
    model.save_pretrained("./outputs/final_model/")
    tokenizer.save_pretrained("./outputs/final_model/")

    print("\n Training completed and model saved successfully!")

if __name__ == "__main__":
    main()
