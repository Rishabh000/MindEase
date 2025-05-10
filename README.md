# MindEase: Emotion-Aware Chatbot
A lightweight chatbot application that detects user emotions from text using a fine-tuned DistilBERT model and generates empathetic responses. Built using FastAPI, deployed via Docker and AWS.

🚀 Features
🎭 Emotion detection using fine-tuned BERT (DistilBERT) model

🤖 Context-aware chatbot responses with emojis

💬 Lightweight frontend interface (HTML + JS)

🐳 Dockerized for easy deployment

☁️ EC2-hosted with public access

🧱 Tech Stack
Frontend: HTML, JavaScript (fetch API)

Backend: FastAPI, Python

Model: Hugging Face Transformers (DistilBERT)

Deployment: Docker, AWS EC2

📁 Project Structure
bash
Copy
Edit
MindEase/
├── app/
│   ├── chatbot.py               # FastAPI backend
│   ├── emotion_model.py         # Emotion prediction logic
│   ├── index.html               # Frontend chat interface
│   └── requirements.txt         # Python dependencies
├── model/
│   ├── config.json              # Tokenizer + model config
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── model.safetensors        # Fine-tuned model weights
├── Dockerfile                   # Docker build instructions
└── README.md
🔧 Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Rishabh000/MindEase.git
cd MindEase
2. Build and Run with Docker
bash
Copy
Edit
docker build -t chatbot-app .
docker run -d -p 80:8000 --name chatbot chatbot-app
Access at: http://<your-ec2-ip>/

💬 API Endpoints
Method	Endpoint	Description
GET	/	Loads the chatbot UI
POST	/chat	Accepts a message and returns emotion-based response

Example request:

json
Copy
Edit
POST /chat
{
  "message": "I'm feeling low today..."
}
Example response:

json
Copy
Edit
{
  "input": "I'm feeling low today...",
  "emotion": "sadness",
  "response": "😢 I'm here for you. Would you like to talk about what’s making you feel this way?"
}

Acknowledgments
Transformers by Hugging Face, FastAPI, ClearML for training tracking
