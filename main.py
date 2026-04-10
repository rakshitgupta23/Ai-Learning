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

raw_documents = [
    """Apples are healthy and rich in fiber. They help digestion and improve gut health. 
    Regular consumption of apples may reduce risk of chronic diseases.""",

    """Soft drinks contain high sugar and are unhealthy. They are linked to obesity and diabetes. 
    Drinking too many sugary beverages can harm your body.""",

    """Exercise improves cardiovascular health. It strengthens the heart and improves blood circulation. 
    Regular physical activity reduces risk of heart disease."""
]

def chunk_text(text, chunk_size=8, overlap=3):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

documents = []

for doc in raw_documents:
    documents.extend(chunk_text(doc))

def get_embedding(text: str):
    response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config={
        "output_dimensionality": 768
    }
)
    return response.embeddings[0].values

doc_embeddings = None

def init_embeddings():
    global doc_embeddings
    if doc_embeddings is None:
        doc_embeddings = [(doc, get_embedding(doc)) for doc in documents]

import math

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b)

def retrieve(query: str):
    init_embeddings()
    query_emb = get_embedding(query)
    scores = []

    for doc, emb in doc_embeddings:
        score = cosine_similarity(query_emb, emb)
        scores.append((doc, score))

    # sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    print(scores)

    top_docs = [doc for doc, score in scores if score > 0.6][:3]  # threshold and top-k

    print("Retrieved documents:", top_docs)
    return top_docs

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    text = request.text
    # Step 1: validate input
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    attempt = 0
    print("Received text to analyze:", text)
    for attempt in range(2):
    # Step 2: call llm
        try:
            rewritten_query = rewrite_query(text)
            queries = generate_queries(rewritten_query)

            all_docs = []

            for q in queries:
                docs = retrieve(q)
                all_docs.extend(docs)

            print("All retrieved docs:", all_docs)

            unique_docs = list(set(all_docs))[:3]
            print("Unique docs:", unique_docs)

            raw=call_llm(text, unique_docs)
        # Step 3: parse output
            cleaned = clean_json(raw)
            data = json.loads(cleaned)
        # Step 4: return response
            return AnalyzeResponse(**data)
        except ServerError:
            print(f"Retrying due to 503... attempt {attempt+1}")
            time.sleep(2)
        except Exception as e:
            print(f"Parsing failed attempt {attempt+1}")
            print(f"Error: {e}")
    return AnalyzeResponse(summary="Unable to analyze text", confidence="low", reason="LLM call failed after retries or output was unparsable")

def clean_json(raw: str):
    raw = raw.strip()

    # remove markdown ```json ```
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    return raw

def rewrite_query(query: str):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"""
Rewrite the following user query to make it more clear and detailed 
for semantic search. Do NOT answer it.

Query:
{query}

Return only the rewritten query.
"""
    )

    rewritten = response.text.strip()
    print("Rewritten query:", rewritten)
    return rewritten

def generate_queries(query: str):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"""
Generate 3 DIFFERENT types of queries to improve retrieval.

Rules:
- Do NOT answer
- Each query must cover a DIFFERENT intent:
  1. Nutritional aspect
  2. Physical performance / gym impact
  3. General health effects
- Keep them short

Query:
{query}

Return a list(one query per line).
"""
    )

    queries = response.text.strip().split("\n")
    queries = [q.strip("- ").strip() for q in queries if q.strip()]

    print("Generated queries:", queries)
    return queries

def call_llm(text: str, context: list[str]):
    try:
        # Step 1: retrieve relevant docs
        context = "\n".join(context)

        # Step 2: call LLM
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
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
