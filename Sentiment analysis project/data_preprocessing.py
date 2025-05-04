# data_preprocessing.py

import pandas as pd
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import os
from collections import Counter

def main():
    # Step 1: Load the DailyDialog dataset
    dailydialog = load_dataset("daily_dialog")

    # Step 2: Flatten the DailyDialog dataset
    emotion_mapping = ["no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    flattened_data = []
    for split in ['train', 'validation', 'test']:
        for item in dailydialog[split]:
            dialog = item['dialog']
            emotions = item['emotion']
            for sentence, emotion in zip(dialog, emotions):
                emotion_label = emotion_mapping[emotion]  # Convert integer emotion to string label
                flattened_data.append({"text": sentence.strip(), "emotion": emotion_label})

    flattened_df = pd.DataFrame(flattened_data)
    flattened_df['emotion'] = flattened_df['emotion'].astype(str)  # Force all emotions as strings

    print(f"[INFO] Flattened data size: {len(flattened_df)}")
    print(flattened_df.head())

    # Step 3: Load the GoEmotions dataset
    goemotions = load_dataset("go_emotions")

    label_mapping_go = goemotions['train'].features['labels'].feature.names

    goemotions_data = []
    for split in ['train', 'validation', 'test']:
        for sample in goemotions[split]:
            text = sample['text']
            if sample['labels']:
                emotion = label_mapping_go[sample['labels'][0]]
                goemotions_data.append({"text": text, "emotion": emotion})

    goemotions_df = pd.DataFrame(goemotions_data)
    print(f"[INFO] GoEmotions data loaded with size: {len(goemotions_df)}")
    print(goemotions_df.head())

    # Step 4: Map GoEmotions fine-grained labels into broad categories
    emotion_mapping_go_to_broad = {
        "admiration": "happiness", "amusement": "happiness", "approval": "happiness",
        "caring": "happiness", "desire": "happiness", "excitement": "happiness",
        "gratitude": "happiness", "joy": "happiness", "love": "happiness",
        "optimism": "happiness", "pride": "happiness", "relief": "happiness",
        "anger": "anger", "annoyance": "anger",
        "disappointment": "sadness", "disapproval": "sadness", "remorse": "sadness",
        "grief": "sadness", "sadness": "sadness", "embarrassment": "sadness",
        "fear": "fear", "nervousness": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
        "neutral": "no_emotion", "confusion": "no_emotion", "curiosity": "no_emotion", "realization": "no_emotion"
    }

    goemotions_df['emotion'] = goemotions_df['emotion'].map(emotion_mapping_go_to_broad)
    goemotions_df = goemotions_df.dropna(subset=['emotion'])  # Drop rows with NaN emotions

    # Step 5: Combine the two datasets
    combined_df = pd.concat([flattened_df, goemotions_df], ignore_index=True)
    combined_df = combined_df[combined_df['emotion'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

    # Step 6: Save the full combined dataset
    os.makedirs("./data", exist_ok=True)
    combined_df.to_csv("./data/combined_emotion_data.csv", index=False)
    print("[OK] Combined dataset saved successfully!")

    # Step 7: Sample 2000 examples for training
    sampled_df = combined_df.sample(n=2000, random_state=42).reset_index(drop=True)
    sampled_df.to_csv("./data/sampled_combined_emotion_data.csv", index=False)
    print("[OK] Sampled dataset (2000 samples) saved successfully!")
    emotion_distribution = sampled_df['emotion'].value_counts()

    print("\n[INFO] Emotion Distribution in Sampled Dataset (2000 examples):")
    print(emotion_distribution)


    # Step 8: Create label mapping and save
    unique_labels = sampled_df['emotion'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    with open("./data/label_mapping.json", "w") as f:
        json.dump(label_mapping, f)
    print("[OK] Label mapping saved successfully!")

    # Step 9: Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained("./data/tokenizer/")
    print("[OK] Tokenizer saved successfully!")

    # Step 10: Final Verification
    print("\n[INFO] Verification Check:")
    print("- Sampled data shape:", sampled_df.shape)
    print("- Unique labels:", unique_labels)
    print("- Label mapping:", label_mapping)

if __name__ == "__main__":
    main()
