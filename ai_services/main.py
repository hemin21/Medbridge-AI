"""
MedBridge AI — AI Services
Medical Document Processing, Summarization, Risk Prediction & Drug Interaction Engine
Powered by Google Gemini
"""
import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="../backend/.env")

app = FastAPI(
    title="MedBridge AI Services",
    description="AI-powered medical processing microservices using Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gemini Configuration ──────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
# If no key is provided, we will fallback to standard logic or return an error depending on the endpoint


# ── Models ──────────────────────────────────────────────────────────

class SymptomInput(BaseModel):
    name: str
    severity: int  # 1-10
    duration: Optional[str] = None

class VitalsInput(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None

class RiskPredictionRequest(BaseModel):
    symptoms: List[SymptomInput]
    vitals: Optional[VitalsInput] = None
    chronic_conditions: Optional[List[str]] = []
    age: Optional[int] = None

class MedicationInput(BaseModel):
    name: str
    dosage: Optional[str] = None

class DrugInteractionRequest(BaseModel):
    medications: List[MedicationInput]

class SummarizationRequest(BaseModel):
    patient_name: str
    blood_group: Optional[str] = None
    allergies: Optional[List[str]] = []
    chronic_diseases: Optional[List[str]] = []
    medications: Optional[List[str]] = []
    recent_symptoms: Optional[List[str]] = []


# ── Health Check ────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    ai_status = "active" if GEMINI_API_KEY else "missing_key"
    return {"status": "ok", "service": "MedBridge AI Services", "gemini_status": ai_status}


# ── OCR / Document Processing ──────────────────────────────────────

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """
    Process an uploaded medical document using Gemini 1.5.
    Extracts text, recognizes entities, and structuring to JSON.
    """
    contents = await file.read()
    
    if not GEMINI_API_KEY:
        # Fallback simulation
        return {
            "status": "processed",
            "filename": file.filename,
            "raw_text": "[SIMULATED - MISSING GEMINI_API_KEY] Blood Test Results...",
            "entities": [
                {"type": "diagnosis", "value": "Type 2 Diabetes", "confidence": 0.9}
            ],
            "structured_data": {"document_type": "lab_report"},
            "summary": "Simulated output. Please provide a Gemini API Key in backend/.env."
        }

    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = """
        You are a medical document AI. Analyze the attached document. 
        Extract the text and strictly return the result as a raw JSON object string ONLY (do not use markdown formatting blocks like ```json).
        The JSON must match this structure exactly:
        {
            "raw_text": "A brief textual extraction of the key findings",
            "entities": [
                {"type": "medication or diagnosis or lab_result", "value": "String", "confidence": 0.0-1.0}
            ],
            "structured_data": {
                "document_type": "prescription or lab_report or discharge_summary",
                "date": "YYYY-MM-DD or unknown",
                "provider": "Hospital/Doctor name or unknown"
            },
            "summary": "1-2 sentence medical summary of the document"
        }
        """
        
        response = model.generate_content([
            {"mime_type": file.content_type, "data": contents}, 
            prompt
        ])
        
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        result = json.loads(text)
        result["status"] = "processed"
        result["filename"] = file.filename
        return result
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return {"error": "AI Document Processing Failed", "details": str(e)}


# ── Risk Prediction ────────────────────────────────────────────────

@app.post("/predict-risk")
def predict_risk(request: RiskPredictionRequest):
    """
    AI risk prediction based on symptoms and vitals using Gemini.
    """
    if not GEMINI_API_KEY:
        return {
            "risk_score": 50,
            "urgency": "urgent",
            "risk_factors": ["Simulated due to missing API key"],
            "recommendations": ["Please configure GEMINI_API_KEY"]
        }
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an emergency medical triage AI. Analyze the patient data and return a JSON object (no markdown).
        Patient data: {request.model_dump_json()}
        
        Calculate a risk score from 0 to 100 based on the severity of the symptoms and vitals.
        Determine urgency: 'routine', 'soon', 'urgent', or 'emergency'.
        Provide risk factors and recommendations.
        
        Return exactly this JSON:
        {{
            "risk_score": INT,
            "urgency": "STRING",
            "risk_factors": ["LIST OF RISK STRINGS"],
            "recommendations": ["LIST OF RECOMMENDATION STRINGS"]
        }}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        result = json.loads(text)
        return result
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return {"error": "Risk prediction failed", "details": str(e)}


# ── Drug Interaction Engine ────────────────────────────────────────

@app.post("/check-interactions")
def check_drug_interactions(request: DrugInteractionRequest):
    """Check for known drug interactions using Gemini."""
    if not GEMINI_API_KEY:
        return {"safe": True, "interactions": [{"description": "Simulated safe due to missing API key"}]}
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an advanced pharmacology AI. Analyze these medications for interactions:
        {request.model_dump_json()}
        
        Return ONLY a JSON list (no markdown) of interactions found, or an empty list. Each object must be:
        {{
            "drug1": "Name",
            "drug2": "Name",
            "severity": "severe or moderate or mild",
            "description": "Clinical risk description"
        }}
        """
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        interactions = json.loads(text)
        
        return {
            "total_medications": len(request.medications),
            "interactions_found": len(interactions),
            "interactions": interactions,
            "safe": len(interactions) == 0,
        }
    except Exception as e:
        return {"error": "Drug interaction check failed", "details": str(e)}


# ── Medical Summarization ──────────────────────────────────────────

@app.post("/summarize")
def summarize_patient(request: SummarizationRequest):
    """Generate a concise medical summary for a patient using Gemini."""
    if not GEMINI_API_KEY:
        return {"summary": "Simulated summary missing API key.", "tldr": "Missing API Key"}
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a medical AI assistant. Write a clear, concise medical summary for this patient profile:
        {request.model_dump_json()}
        
        Return ONLY a JSON dictionary:
        {{
            "summary": "Detailed professional paragraph summary of their current health status and profile.",
            "tldr": "1 sentence ultra-short summary highlighting key risks or conditions."
        }}
        """
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        return json.loads(text)
    except Exception as e:
        return {"error": "Summarization failed", "details": str(e)}


# ── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
