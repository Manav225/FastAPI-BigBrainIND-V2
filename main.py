"""
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import json
import os

# ----------------------------
# Authenticate with Hugging Face
# ----------------------------
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN not set. Please set it before running.")

# ----------------------------
# Load label map
# ----------------------------
with open("id2label.json", "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# ----------------------------
# Load tokenizer & model ONCE
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI()

class Complaint(BaseModel):
    text: str
    threshold: float = 0.5

@app.post("/predict")
def predict(data: Complaint):
    # Tokenize input
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # Multi-label probabilities
    probs = torch.sigmoid(outputs.logits)

    # Apply threshold
    predicted_indices = (probs > data.threshold).nonzero(as_tuple=True)[1].tolist()
    predicted_labels = [id2label[i] for i in predicted_indices]

    return {
        "complaint": data.text,
        "predicted_labels": predicted_labels,
        "probabilities": probs.cpu().numpy().tolist()
    }
"""

#Working Code
"""
import os
import json
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Authenticate with Hugging Face
# ----------------------------
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN not set. Please set it before running.")

# ----------------------------
# Load label map
# ----------------------------
with open("id2label.json", "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# ----------------------------
# Load tokenizer & model ONCE
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI()

# Enable CORS for browser access
origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Complaint(BaseModel):
    text: str
    threshold: float = 0.5

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

@app.post("/predict")
def predict(data: Complaint):
    # Tokenize input
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # Multi-label probabilities
    probs = torch.sigmoid(outputs.logits)

    # Apply threshold
    predicted_indices = (probs > data.threshold).nonzero(as_tuple=True)[1].tolist()
    predicted_labels = [id2label[i] for i in predicted_indices]

    return {
        "complaint": data.text,
        "predicted_labels": predicted_labels,
        "probabilities": probs.cpu().numpy().tolist()
    }

# ----------------------------
# Run Uvicorn when executed directly
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
"""
import os
import json
import torch
import uvicorn
import re
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from fastapi.middleware.cors import CORSMiddleware
from sarvamai import SarvamAI   # NEW: import Sarvam

# ----------------------------
# Authenticate with Hugging Face
# ----------------------------
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN not set. Please set it before running.")

# ----------------------------
# Load label map
# ----------------------------
with open("id2label.json", "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# ----------------------------
# Load tokenizer & model ONCE
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# Init Sarvam Client
# ----------------------------
sarvam_api_key = os.getenv("SARVAM_API_KEY")
if not sarvam_api_key:
    raise EnvironmentError("SARVAM_API_KEY not set.")
sarvam_client = SarvamAI(api_subscription_key=sarvam_api_key)

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI()

origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Complaint(BaseModel):
    text: str
    threshold: float = 0.5

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

@app.post("/predict")
def predict(data: Complaint):
    # --- Step 1: HF Classification ---
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)
    predicted_indices = (probs > data.threshold).nonzero(as_tuple=True)[1].tolist()
    predicted_labels = [id2label[i] for i in predicted_indices]

    # --- Step 2: Sarvam Reasoning ---
    legal_reasoning = {}
    if predicted_labels:
        prompt_template = """
        You are a legal reasoning assistant trained on Indian Penal Code (IPC).
        Task:
        Complaint: {complaint}
        ONLY use the following IPC sections predicted by the model: {labels}
        Ignore any other IPC sections mentioned in the complaint text.

        For each of the given IPC sections:
        - State the legal elements.
        - Map facts from the complaint.
        - State strength (Strong/Weak/Indeterminate).
        - Give short prima facie conclusion.

        Output must be JSON only (no markdown, no code block)
        """
        user_prompt = prompt_template.format(
            complaint=data.text,
            labels=", ".join(predicted_labels)
        )

        sarvam_response = sarvam_client.chat.completions(
            messages=[{"role": "user", "content": user_prompt}]
        )

        #raw_output = sarvam_response["choices"][0]["message"]["content"]
        raw_output = sarvam_response.choices[0].message.content  # ✅ fixed

        # --- Sanitize JSON ---
        try:
            # remove markdown code fences if present
            cleaned = re.sub(r"^```json|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
            legal_reasoning = json.loads(cleaned)
        except Exception:
            legal_reasoning = {"error": "Could not parse Sarvam response", "raw": raw_output}

    # --- Step 3: Return Combined Output ---
    return {
        "complaint": data.text,
        "predicted_labels": predicted_labels,
        "probabilities": probs.cpu().numpy().tolist(),
        "legal_reasoning": legal_reasoning
    }
# ----------------------------
# Run Uvicorn when executed directly
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
