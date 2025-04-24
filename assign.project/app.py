from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import re

app = FastAPI()

# Load the trained model
model = joblib.load("rf_email_classifier.pkl")

# PII Masking function
def mask_pii_and_track(text):
    masked_text = text
    masked_entities = []
    patterns = {
        'full_name': r'\b([A-Z][a-z]+\s[A-Z][a-z]+)\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone_number': r'\b(\+91[-\s]?|0)?[6-9]\d{9}\b',
        'dob': r'\b(0?[1-9]|[12][0-9]|3[01])[- /.](0?[1-9]|1[012])[- /.](\d{4})\b',
        'aadhar_num': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'credit_debit_no': r'\b(?:\d[ -]*?){13,16}\b',
        'cvv_no': r'\b\d{3}\b',
        'expiry_no': r'\b(0[1-9]|1[0-2])\/\d{2,4}\b'
    }

    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            original_entity = match.group(0)
            masked_entities.append({
                'position': [start, end],
                'classification': entity_type,
                'entity': original_entity
            })
            masked_text = re.sub(re.escape(original_entity), f'[{entity_type}]', masked_text, count=1)

    return masked_text, masked_entities

# Subcategory assignment (reuse your function here)
def assign_subcategory(text, category):
    text = text.lower()
    if category == 'Incident':
        if re.search(r'\b(software|application|program)\b', text):
            return 'Software Malfunction'
        elif re.search(r'\b(access|login|authentication)\b', text):
            return 'Access Issues'
        elif re.search(r'\b(error|failure|crash|bug)\b', text):
            return 'System Errors'
        elif re.search(r'\b(troubleshoot|fix|resolve|diagnose)\b', text):
            return 'Technical Troubleshooting'
        elif re.search(r'\b(data|information|corrupt|missing)\b', text):
            return 'Data Issues'
        else:
            return 'General Incident'
    elif category == 'Request':
        if re.search(r'\b(information|details|clarification|query)\b', text):
            return 'Request for Information'
        elif re.search(r'\b(account|profile|update|modify|change)\b', text):
            return 'Account/Profile Update'
        elif re.search(r'\b(integration|connect|api|access)\b', text):
            return 'Integration/Feature Access'
        elif re.search(r'\b(contact|phone|email|reach)\b', text):
            return 'Contact Request'
        else:
            return 'General Request'
    elif category == 'Problem':
        if re.search(r'\b(recurring|repeat|consistent)\b', text):
            return 'Recurring Technical Failure'
        elif re.search(r'\b(bug|glitch|defect)\b', text):
            return 'Bug/Glitches'
        elif re.search(r'\b(escalate|urgent|critical|priority)\b', text):
            return 'Escalated Issue'
        elif re.search(r'\b(data|database|loss|integrity)\b', text):
            return 'Data-Related Problem'
        else:
            return 'General Problem'
    elif category == 'Change':
        if re.search(r'\b(request|modify|adjust|alter|configure)\b', text):
            return 'Configuration Change Request'
        elif re.search(r'\b(tool|system|update|upgrade|patch)\b', text):
            return 'System/Tool Update'
        elif re.search(r'\b(enable|activate|disable|permission|role)\b', text):
            return 'Feature Enablement/Access Rights'
        elif re.search(r'\b(environment|setup|deployment|infrastructure)\b', text):
            return 'Environment/Setup Change'
        else:
            return 'General Change Request'
    return 'Uncategorized Subtype'

# Request and response models
# 🧩 Updated response model
class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class ClassificationResult(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

# 🎯 Updated endpoint
@app.post("/classify", response_model=ClassificationResult)
def classify_email(input_data: EmailInput):
    try:
        masked_email, masked_entities = mask_pii_and_track(input_data.email_text)
        predicted_category = model.predict([masked_email])[0]

        return ClassificationResult(
            input_email_body=input_data.email_text,
            list_of_masked_entities=masked_entities,
            masked_email=masked_email,
            category_of_the_email=predicted_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

