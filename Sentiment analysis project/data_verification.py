import pandas as pd
from collections import Counter

def verify_sampled_data_distribution(csv_path):
    df = pd.read_csv(csv_path)
    label_counts = Counter(df['emotion'])
    
    print("\n✅ Sampled Data Label Distribution (after preprocessing and saving):")
    for label, count in label_counts.items():
        print(f"{label}: {count} samples")

# Call after saving the CSV
verify_sampled_data_distribution("data/sampled_combined_emotion_data.csv")   # or whatever path you saved
