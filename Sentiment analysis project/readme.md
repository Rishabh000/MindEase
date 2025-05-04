This project builds a broad emotion detection model by combining multiple datasets and implements a ClearML-powered MLOps pipeline for:

Dataset management & versioning

Training and Fine-tuning

Model version tracking and loss/accuracy visualization

ClearML allows full experiment reproducibility, dataset version control, and fine-tuning with minimal manual intervention.

Steps to run the code:-
1- Setup virtual environment
    python -m venv venv
    venv\Scripts\activate

2- Install dependencies- pip install -r requirements.txt

3- Configure clearml- clearml-init

4- Preprocess the data
    python data_preprocessing.py
what this does:-
    Downloads and combines DailyDialog + GoEmotions datasets

    Cleans and maps emotion labels

    Saves a sample of 2000 rows for training

    Saves sampled_combined_emotion_data.csv, label_mapping.json, and tokenizer.

5- Create clearML dataset version
    python create_dataset.py
what this does:-
    Uploads the processed dataset to ClearML
    Creates a ClearML versioned dataset entry (for future tracking and fine-tuning)

6- Train the model
    python train.py
what this does:-
    Loads dataset and tokenizer

    Fine-tunes a DistilBERT model on broad emotion categories

    Tracks train loss, validation loss, and accuracy in ClearML Dashboard

    Saves model checkpoints and final model.

7- Fine-tuning on new data
    python fine_tuning.py

Key Features:-
    Dataset Versioning via ClearML

    Automated Model Tracking with plots and artifacts

    Model Fine-tuning Pipeline ready for new data

    Full Reproducibility (ClearML auto-captures all code, dependencies, parameters)

    Minimal Manual Intervention once setup is complete

what you see in the clearml dashboard
Training Loss vs Validation Loss curves

Validation Accuracy curve

GPU/CPU Usage graphs

Logged Metrics (loss, accuracy, learning rate, etc.)

Artifacts: Final model + tokenizer checkpoints

Dataset Versions: Uploaded and managed
