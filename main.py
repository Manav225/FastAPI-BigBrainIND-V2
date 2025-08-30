#Working Code
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
"""
import os
import json
import re
import logging
import torch
import uvicorn
from typing import Any, Dict, List, Tuple, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from fastapi.middleware.cors import CORSMiddleware
from sarvamai import SarvamAI


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Authenticate with Hugging Face
# ----------------------------
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        logger.error(f"Hugging Face login failed: {e}")
else:
    logger.warning("HUGGINGFACE_HUB_TOKEN not set. Model may not load.")


# ----------------------------
# Load label map
# ----------------------------
try:
    with open("id2label.json", "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
except FileNotFoundError:
    logger.error("id2label.json not found. Make sure the file exists.")
    id2label = {}


# ----------------------------
# Load tokenizer & model ONCE
# ----------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained("Manav225/BigBrainLAW-IND", token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    tokenizer = None
    model = None


# ----------------------------
# SarvamAI SDK Configuration
# ----------------------------
sarvam_api_key = os.getenv("SARVAM_API_KEY")
if not sarvam_api_key:
    logger.warning("SARVAM_API_KEY not set. Reasoning will not be available.")
    sarvam_client = None
else:
    try:
        sarvam_client = SarvamAI(api_subscription_key=sarvam_api_key)
        logger.info("SarvamAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SarvamAI client: {e}")
        sarvam_client = None


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()
origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Pydantic Schemas
# ----------------------------
class SectionReasoning(BaseModel):
    elements: List[str] = Field(default_factory=list)
    fact_mapping: List[str] = Field(default_factory=list)
    support_strength: str = ""
    conclusion: str = ""
    raw_text: str = ""
    reasoning_available: bool = True


class Complaint(BaseModel):
    text: str
    threshold: float = 0.5


# ----------------------------
# JSON extraction & parsing helpers
# ----------------------------
def clean_section_label(label: str) -> str:
    if not isinstance(label, str):
        return str(label)
    m = re.search(r"Section\s+(\d+)", label, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"IPC[_\s-]*(\d+)", label, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    m3 = re.search(r"\b(\d{1,4})\b", label)
    if m3:
        return m3.group(1)
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")[:20]


def parse_reasoning_output(raw_output: str) -> Dict[str, SectionReasoning]:
    """Parse the response from SarvamAI Chat Completions API"""
    try:
        # The response might be directly a JSON string or contain JSON
        json_str = raw_output.strip().strip("`").strip()
        
        # Try to find JSON in the response
        json_start = json_str.find('{')
        json_end = json_str.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_str = json_str[json_start:json_end+1]
        
        parsed = json.loads(json_str)
        
        normalized: Dict[str, SectionReasoning] = {}
        for key, val in parsed.items():
            sec = clean_section_label(str(key))
            if isinstance(val, dict):
                normalized[sec] = SectionReasoning(
                    elements=val.get("elements", []),
                    fact_mapping=val.get("fact_mapping", []),
                    support_strength=val.get("support_strength", ""),
                    conclusion=val.get("conclusion", ""),
                    raw_text=raw_output,
                    reasoning_available=True
                )
            else:
                normalized[sec] = SectionReasoning(
                    conclusion=str(val),
                    raw_text=raw_output,
                    reasoning_available=True
                )
        return normalized
    except Exception as e:
        logger.warning(f"Failed to parse JSON output from SarvamAI: {e}")
        return {
            "general": SectionReasoning(
                conclusion=f"Legal reasoning could not be parsed: {raw_output}",
                raw_text=raw_output,
                reasoning_available=False
            )
        }


def call_sarvam_chat_completions(complaint_text: str, predicted_sections: List[str]) -> str:
    """Call SarvamAI Chat Completions API for legal reasoning using official SDK"""
    
    if not sarvam_client:
        raise Exception("SarvamAI client not initialized")
    
    # Create the legal analysis prompt
    prompt = f"""
You are a legal reasoning assistant for the Indian Penal Code (IPC).

Complaint: {complaint_text}

Analyze only these IPC sections: {", ".join(predicted_sections)}

For each given IPC section, provide a concise analysis in JSON format.
The JSON must be a single object where keys are the IPC section numbers and values are objects with these fields:
- "elements": List of key legal elements.
- "fact_mapping": List of facts from the complaint that map to the elements.
- "support_strength": "Strong", "Weak", or "Indeterminate".
- "conclusion": A brief summary of the analysis.

Example JSON format:
{{
  "420": {{
    "elements": ["Deception", "Fraudulent inducement"],
    "fact_mapping": ["Mr. XYZ took money", "Promised a job"],
    "support_strength": "Strong",
    "conclusion": "The complaint outlines clear elements of deception and inducement for financial gain."
  }}
}}

Return ONLY the JSON object, no additional text.
""".strip()
    
    try:
        # Call SarvamAI using the official SDK
        response = sarvam_client.chat.completions(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the content from the response
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        elif hasattr(response, 'content'):
            return response.content
        else:
            logger.warning(f"Unexpected SarvamAI response format: {response}")
            return str(response)
            
    except Exception as e:
        logger.error(f"SarvamAI SDK call failed: {e}")
        raise Exception(f"SarvamAI SDK call failed: {str(e)}")


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"message": "Legal Analysis API is running with SarvamAI SDK!"}


@app.post("/predict")
async def predict(data: Complaint):
    try:
        # Step 1: HuggingFace classification
        if not model or not tokenizer:
            return {
                "complaint": data.text,
                "predicted_labels": [],
                "predicted_labels_normalized": [],
                "probabilities": [],
                "legal_reasoning": {
                    "general": SectionReasoning(
                        conclusion="Model not loaded. Check server logs.",
                        reasoning_available=False
                    ).dict()
                }
            }
        
        inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits)
        predicted_indices = (probs > data.threshold).nonzero(as_tuple=True)[1].tolist()
        
        raw_labels = [id2label.get(i, "Unknown Section") for i in predicted_indices]
        predicted_labels_normalized = [clean_section_label(lbl) for lbl in raw_labels]
        
        # Step 2: SarvamAI legal reasoning using SDK
        legal_reasoning: Dict[str, Any] = {}
        
        if predicted_labels_normalized and sarvam_client:
            try:
                # Call SarvamAI Chat Completions using SDK
                raw_output = call_sarvam_chat_completions(data.text, raw_labels)
                legal_reasoning_models = parse_reasoning_output(raw_output)
                
                # Convert Pydantic models to dicts for JSON response
                legal_reasoning = {sec: reasoning.dict() for sec, reasoning in legal_reasoning_models.items()}
                
            except Exception as e:
                logger.exception("SarvamAI Chat Completions API call failed.")
                legal_reasoning = {
                    "general": SectionReasoning(
                        conclusion=f"Reasoning not available due to API error: {str(e)}",
                        reasoning_available=False
                    ).dict()
                }
        else:
            legal_reasoning = {
                "general": SectionReasoning(
                    conclusion="No IPC sections predicted or SarvamAI client not initialized.",
                    reasoning_available=False
                ).dict()
            }

        # Step 3: Return final response
        return {
            "complaint": data.text,
            "predicted_labels": raw_labels,
            "predicted_labels_normalized": predicted_labels_normalized,
            "probabilities": probs.cpu().numpy().tolist(),
            "legal_reasoning": legal_reasoning,
        }

    except Exception as e:
        logger.exception("Unexpected error in /predict")
        return {
            "complaint": data.text,
            "predicted_labels": [],
            "predicted_labels_normalized": [],
            "probabilities": [],
            "legal_reasoning": {
                "general": SectionReasoning(
                    conclusion=f"Reasoning not available due to internal server error: {str(e)}",
                    reasoning_available=False
                ).dict()
            }
        }


# ----------------------------
# Health check endpoint for debugging
# ----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "sarvam_configured": sarvam_client is not None,
        "device": str(device) if 'device' in globals() else "unknown"
    }


# ----------------------------
# Test SarvamAI SDK connection endpoint
# ----------------------------
@app.get("/test-sarvam")
def test_sarvam():
    if not sarvam_client:
        return {"error": "SarvamAI client not initialized"}
    
    try:
        # Test with a simple question using SDK
        response = sarvam_client.chat.completions(
            messages=[
                {"role": "user", "content": "Who was Aryabhata? Explain in 3 lines."}
            ]
        )
        
        # Extract content from response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        elif hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
            
        return {
            "status": "success",
            "response": content,
            "full_response": str(response)
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }


# ----------------------------
# Test legal analysis endpoint
# ----------------------------
@app.get("/test-legal")
def test_legal():
    if not sarvam_client:
        return {"error": "SarvamAI client not initialized"}
    
    try:
        # Test legal reasoning with sample data
        test_response = call_sarvam_chat_completions(
            "A person stole money from a bank", 
            ["Section 420", "Section 379"]
        )
        return {
            "status": "success",
            "legal_response": test_response
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }


# ----------------------------
# Run Uvicorn when executed directly
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
