import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# 1. Create the app object
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Replace with the domain where your frontend is served
    allow_credentials=True,
    allow_methods=["*"],  # Or restrict to specific methods like ['GET', 'POST']
    allow_headers=["*"],  # Or restrict to specific headers like ['Content-Type']
)


# 2. Load the tokenizer and model properly from a directory or a saved checkpoint
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('/Users/avishekagarwal/Desktop/App')  # Example path to your saved model directory

# 3. Define a Pydantic model for the input data
class PredictionInput(BaseModel):
    text: str

# 4. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 5. Route with a single parameter, returns the parameter within a message
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Avi website': f'{name}'}

# 6. Expose the prediction functionality
@app.post('/predict')
def predict_news(data: PredictionInput):
    try:
        input_text = data.text

        # Tokenize the input text
        inputs = tokenizer(
            input_text, 
            truncation=True, 
            padding='max_length', 
            max_length=42, 
            return_tensors='tf'
        )

        # Get the predictions
        outputs = model(inputs)
        logits = outputs.logits[0]

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits).numpy()

        # Get the predicted label
        predicted_label = np.argmax(probabilities)

        # Create a human-readable label
        prediction = "Fake News" if predicted_label == 0 else "True News"
        return {
            'prediction': prediction,
            'confidence': float(probabilities[predicted_label])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server, use the command:
# uvicorn filename:app --reload
