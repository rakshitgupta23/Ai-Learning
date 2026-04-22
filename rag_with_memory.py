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
memory_embeddings = []

def store_memory_fact(fact: str):
    emb = get_embedding(fact)
    memory_embeddings.append((fact, emb))

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

def retrieve_memory(query: str, top_k=2):
    if not memory_embeddings:
        return []

    query_emb = get_embedding(query)
    scores = []

    for fact, emb in memory_embeddings:
        score = cosine_similarity(query_emb, emb)
        scores.append((fact, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    top_facts = [fact for fact, _ in scores[:top_k]]

    print("Relevant memory:", top_facts)
    return top_facts

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

chat_history = []
user_memory = []

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    text = request.text

    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    print("Received:", text)

    # 1️⃣ Store user message
    chat_history.append({"role": "user", "content": text})

    # 2️⃣ Extract memory (ONLY if meaningful)
    facts = extract_memory(text)
    print("Extracted facts:", facts)

    for fact in facts:
        if fact not in user_memory:
            user_memory.append(fact)
            store_memory_fact(fact)

    print("User memory:", user_memory)
    print("Chat history:", chat_history)

    try:
        # 3️⃣ Rewrite query
        rewritten_query = rewrite_query(text)

        # 4️⃣ Retrieve context
        context = retrieve(rewritten_query)

        relevant_memory = retrieve_memory(text)

        # 5️⃣ Generate answer
        raw = generate_answer(
            query=text,
            context=context,
            history=chat_history,
            memory=relevant_memory
        )

        # 6️⃣ Clean + parse
        cleaned = clean_json(raw)
        data = json.loads(cleaned)

        # 7️⃣ Store assistant response (ONLY summary)
        chat_history.append({
            "role": "assistant",
            "content": data["summary"]
        })

        print("Chat history after response:", chat_history)

        return AnalyzeResponse(**data)

    except Exception as e:
        print("Error:", e)
        return AnalyzeResponse(
            summary="System failed",
            confidence="low",
            reason="Error occurred"
        )

def clean_json(raw: str):
    raw = raw.strip()

    # remove markdown ```json ```
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    return raw

def extract_memory(text: str):
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"""
Extract important long-term user facts.

Rules:
- Only extract stable facts (health, goals, preferences)
- Ignore questions
- Return JSON

Message:
{text}

Return:
{{
  "facts": ["..."]
}}
"""
        )

        cleaned = response.text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        data = json.loads(cleaned)
        return data.get("facts", [])

    except:
        return []

def rewrite_query(query: str):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
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
        model="gemini-3.1-flash-lite-preview",
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

def generate_answer(query: str, context: list[str], history: list[dict], memory: list[str]):
    try:
        context_text = "\n\n".join(context)

        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-5:]]
        )

        memory_text = "\n".join(memory)

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"""
User Memory:
{memory_text}

Conversation History:
{history_text}

Context:
{context_text}

Question:
{query}

Instructions:
- Use context if available
- Use memory if relevant
- If unsure, say low confidence

Return ONLY JSON:
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
        print("LLM Error:", e)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",       
        host="127.0.0.1",
        port=8000,
        reload=True        
    )
