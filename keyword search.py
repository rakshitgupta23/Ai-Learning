import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import os
from google import genai
from google.api_core.exceptions import ServerError
import json

load_dotenv()


class AnalyzeResponse(BaseModel):
    summary: str
    confidence: Literal["low", "medium", "high"]
    reason: str


class AnalyzeRequest(BaseModel):
    text: str




app = FastAPI()


# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

documents = [
    "Apples are healthy and rich in fiber.",
    "Soft drinks contain high sugar and are unhealthy.",
    "Exercise improves cardiovascular health."
]

def retrieve(query: str):
    query = query.lower()

    synonyms = {
        "coke": "soft drinks",
        "soda": "soft drinks"
    }

    for key in synonyms:
        if key in query:
            query += " " + synonyms[key]

    query_words = query.split()
    results = []

    for doc in documents:
        doc_words = doc.lower().split()

        if any(word in doc_words for word in query_words):
            results.append(doc)

    print("Retrieved documents:", results)
    return results[:2]

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    text = request.text
    # Step 1: validate input
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    attempt = 0
    for attempt in range(2):
    # Step 2: call llm
        try:
            raw=call_llm(text)
        # Step 3: parse output
            cleaned = clean_json(raw)
            data = json.loads(cleaned)
        # Step 4: return response
            return AnalyzeResponse(**data)
        except ServerError:
            print(f"Retrying due to 503... attempt {attempt+1}")
            time.sleep(2)
        except Exception:
            print(f"Parsing failed attempt {attempt+1}")
    return AnalyzeResponse(summary="Unable to analyze text", confidence="low", reason="LLM call failed after retries or output was unparsable")

def clean_json(raw: str):
    raw = raw.strip()

    # remove markdown ```json ```
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    return raw

def call_llm(text: str):
    try:
        # Step 1: retrieve relevant docs
        context = "\n".join(retrieve(text))

        # Step 2: call LLM
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
Use the following context to answer:

Context:
{context}

Question:
{text}

Return ONLY valid JSON:
{{
  "summary": "...",
  "confidence": "low | medium | high",
  "reason": "..."
}}
"""
        )

        print(response.text)
        return response.text

    except Exception as e:
        print("Gemini Error:", e)
        raise



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",       
        host="127.0.0.1",
        port=8000,
        reload=True        
    )
