import json
import re
import boto3
import numpy as np
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List

from langchain_aws import BedrockEmbeddings, ChatBedrock

from config import (
    AWS_REGION,
    BUCKET_NAME,
    EMBEDDINGS_INDEX_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K_RESULTS,
    SIMILARITY_HIGH_THRESHOLD,
    SIMILARITY_MEDIUM_THRESHOLD,
)


s3_client = boto3.client("s3")
_cached_index = None      # Embeddings index (kept in memory only)
_embeddings_model = None  # Titan embeddings model
_llm = None               # Claude LLM instance


def get_embeddings_model():
    """Load Titan Embeddings model"""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = BedrockEmbeddings(
            model_id=EMBEDDING_MODEL,
            region_name=AWS_REGION
        )
    return _embeddings_model


def get_llm():
    """Load Claude model"""
    global _llm
    if _llm is None:
        _llm = ChatBedrock(
            model_id=LLM_MODEL,
            region_name=AWS_REGION,
            model_kwargs={
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
        )
    return _llm


def load_index_into_memory():
    """
    Loads the embeddings index from S3 on the FIRST request (cold start).
    Stores it IN MEMORY for all subsequent invocations.
    """
    global _cached_index

    if _cached_index is not None:
        return _cached_index  # warm start → FAST

    print("Downloading embeddings index from S3...")

    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=EMBEDDINGS_INDEX_KEY)
    _cached_index = json.loads(obj["Body"].read().decode("utf-8"))

    return _cached_index


def detect_query_type(query: str) -> Tuple[str, Optional[str]]:
    patterns = [
        r'(?:chapter|standard|section)\s*([A-Z]{2,3}[.-]\d+)',
        r'show me ([A-Z]{2,3}[.-]\d+)',
        r'give me.*([A-Z]{2,3}[.-]\d+)',
    ]

    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            return "citation", m.group(1).upper().replace("-", ".")
    return "question", None


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    v1 = np.array(v1)
    v2 = np.array(v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def vector_search(query_embedding, chunks, top_k=TOP_K_RESULTS):
    scored = []

    for ch in chunks:
        emb = ch.get("embedding")
        if emb:
            sim = cosine_similarity(query_embedding, emb)
            scored.append((sim, ch))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [{"similarity": s, "chunk": c} for s, c in scored[:top_k]]



def handle_citation(chunks, chapter_id, query):
    for ch in chunks:
        if ch["metadata"]["chapter"] == chapter_id:
            return {
                "query": query,
                "query_type": "citation",
                "chapter": chapter_id,
                "exact_text": ch["text"],
                "metadata": ch["metadata"],
                "chunk_id": ch["chunk_id"],
                "timestamp": datetime.utcnow().isoformat()
            }

    return {
        "query": query,
        "query_type": "citation",
        "chapter": chapter_id,
        "found": False,
        "error": f"Chapter {chapter_id} not found in NIAHO standards."
    }



def handle_question(query, chunks):
    emb_model = get_embeddings_model()
    llm = get_llm()

    # Step 1 — embed the user query
    query_embedding = emb_model.embed_query(query)

    # Step 2 — vector search
    results = vector_search(query_embedding, chunks, TOP_K_RESULTS)

    # Step 3 — build RAG context
    context = "\n\n".join(
        f"[{r['chunk']['metadata']['chapter']}]\n{r['chunk']['text']}"
        for r in results
    )

    # Step 4 — build prompt
    prompt = f"""
        Use ONLY the context below to answer the user's question.
        Cite specific chapter IDs.

        Context:
        {context}

        Question: {query}

        Answer:
        """

    # Step 5 — call Claude
    reply = llm.invoke(prompt).content

    # Step 6 — confidence scoring
    top_sim = results[0]["similarity"] if results else 0
    confidence = (
        "high" if top_sim >= SIMILARITY_HIGH_THRESHOLD else
        "medium" if top_sim >= SIMILARITY_MEDIUM_THRESHOLD else
        "low"
    )

    return {
        "query": query,
        "query_type": "question",
        "answer": reply,
        "citations": [
            {
                "chapter": r["chunk"]["metadata"]["chapter"],
                "section": r["chunk"]["metadata"]["section"],
                "similarity": round(r["similarity"], 3)
            }
            for r in results
        ],
        "confidence": confidence
    }


def query_handler(query: str):
    index = load_index_into_memory()
    chunks = index["chunks"]

    qtype, chapter_id = detect_query_type(query)

    if qtype == "citation":
        return handle_citation(chunks, chapter_id, query)

    return handle_question(query, chunks)


def lambda_handler(event, context):
    try:
        query = event.get("query")
        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "query is required"})
            }

        response = query_handler(query)

        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
