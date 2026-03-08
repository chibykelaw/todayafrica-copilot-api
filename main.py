import os
from typing import List, Optional

import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DB_URL or not OPENAI_API_KEY:
    raise ValueError("Missing SUPABASE_DB_URL or OPENAI_API_KEY in .env")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60,
    max_retries=5,
)

app = FastAPI(title="Today Africa Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://todayafrica.co", "https://www.todayafrica.co"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = psycopg2.connect(DB_URL)
conn.autocommit = True


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = None
    article_url: Optional[str] = None


class SourceItem(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


SYSTEM_PROMPT = """
You are Today Africa AI, an editorial intelligence assistant for Today Africa.

Your job is to answer clearly, professionally, and concisely using the provided Today Africa article context.

Rules:
- Write in plain, polished English.
- Do not use markdown headings like ###.
- Do not use asterisks for bold.
- Keep answers coherent and well-structured.
- Prefer 2 short paragraphs or 3 concise bullet-style points in plain text.
- Only mention ideas supported by the provided context.
- If the question is not well supported by the retrieved context, say so briefly and answer cautiously.
- Do not invent facts, article titles, founders, or URLs.
- When relevant, synthesize the main insight across the retrieved articles instead of dumping generic advice.
"""

def embed_query(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve_relevant_chunks(query_embedding, limit=6):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ac.content,
                a.title,
                a.url
            FROM article_chunks ac
            JOIN articles a ON a.id = ac.article_id
            ORDER BY ac.embedding <-> %s::vector
            LIMIT %s
            """,
            (query_embedding, limit)
        )
        rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            "content": row[0],
            "title": row[1],
            "url": row[2]
        })
    return results


def build_context(chunks):
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {idx}] Title: {chunk['title']}\n"
            f"URL: {chunk['url']}\n"
            f"Content:\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(message: str, history: Optional[List[HistoryItem]], context: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for item in history[-6:]:
            messages.append({"role": item.role, "content": item.content})

    messages.append({
        "role": "system",
        "content": f"Use this Today Africa context when answering:\n\n{context}"
    })

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content


def dedupe_sources(chunks):
    seen = set()
    sources = []

    for chunk in chunks:
        key = (chunk["title"], chunk["url"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": chunk["title"],
                "url": chunk["url"]
            })

    return sources[:4]


def log_query(question: str, answer: str, article_url: Optional[str]):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO copilot_queries (question, answer, source_url)
            VALUES (%s, %s, %s)
            """,
            (question, answer, article_url)
        )


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Today Africa Copilot API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query_embedding = embed_query(request.message)
    chunks = retrieve_relevant_chunks(query_embedding, limit=8)
    context = build_context(chunks)
    answer = generate_answer(request.message, request.history, context)
    sources = dedupe_sources(chunks)

    log_query(request.message, answer, request.article_url)

    return ChatResponse(
        answer=answer,
        sources=[SourceItem(**source) for source in sources]
    )