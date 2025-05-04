FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and supporting code
COPY app/index.html .
COPY app/chatbot.py .
COPY app/emotion_model.py .

# Copy the model
COPY model/ ./model/

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
