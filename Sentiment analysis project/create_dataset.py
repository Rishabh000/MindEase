# create_dataset.py
from clearml import Dataset

def main():
    # Create a ClearML Dataset
    dataset = Dataset.create(
        dataset_name="EmotionDetectionDataset",
        dataset_project="Emotion Detection",
        dataset_tags=["v1", "broad_emotions"]
    )

    # Add your preprocessed CSV
    dataset.add_files(path="data/sampled_combined_emotion_data.csv")

    # Upload dataset to ClearML server
    dataset.upload()

    # Finalize and commit
    dataset.finalize()

    print(f"✅ Dataset created successfully! Dataset ID: {dataset.id}")

if __name__ == "__main__":
    main()
