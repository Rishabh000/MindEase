from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Define label mapping
labels = ["sadness", "happiness", "disgust", "anger", "fear", "surprise", "no_emotion"]

# Correct and cross-platform-safe path
model_dir = "./model"

# Load tokenizer and model once (at module level)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

model.eval()  # Set to eval mode for inference

# Prediction function
def predict_emotion(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  # Disable gradient calculation for faster inference
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return labels[prediction]
