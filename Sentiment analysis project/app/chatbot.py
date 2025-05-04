from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from emotion_model import predict_emotion

app = FastAPI()

# Serve index.html on root path
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse("index.html")

# Generate response from emotion
def generate_response(emotion):
    base = f"I sense that you are feeling {emotion}. "
    return {
        "happiness": base + "That's wonderful! Want to share what made you happy?",
        "sadness": base + "I'm here for you. Would you like to talk about what's making you feel this way?",
        "anger": base + "It's okay to feel angry. Let's discuss what's bothering you.",
        "fear": base + "It sounds like something is worrying you. Feel free to share.",
        "disgust": base + "That must have been unpleasant. You can tell me more if you want.",
        "surprise": base + "Wow, that sounds unexpected! Want to share more?"
    }.get(emotion, base + "I'm here to listen whenever you want to share something.")

# POST endpoint for chatbot
@app.post("/chat")
async def get_chatbot_response(request: Request):
    data = await request.json()
    user_input = data.get("message")

    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'message' in request")

    emotion = predict_emotion(user_input)
    response = generate_response(emotion)
    return {"input": user_input, "response": response}
