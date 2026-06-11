import logging
import time
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import anthropic
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

_retrieval_engine = None
_llm_client = None
_db_session_factory = None

SYSTEM_PROMPT = (
    "You are a helpful enterprise knowledge assistant. "
    "Answer questions accurately based on the provided context. "
    "Always cite your sources."
)


class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class QueryFilters(BaseModel):
    source_type: Optional[str] = None
    date_range: Optional[DateRange] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[QueryFilters] = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceInfo(BaseModel):
    index: int
    source: str
    score: float


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: List[SourceInfo]
    confidence: float
    latency_ms: float
    model: str


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class QueryHistoryItem(BaseModel):
    query_id: str
    question: str
    answer: str
    created_at: datetime
    rating: Optional[int] = None


def _build_pinecone_filters(filters: Optional[QueryFilters]) -> Optional[Dict[str, Any]]:
    if filters is None:
        return None
    pine_filter: Dict[str, Any] = {}
    if filters.source_type:
        pine_filter["source_type"] = {"$eq": filters.source_type}
    if filters.date_range:
        date_filter: Dict[str, Any] = {}
        if filters.date_range.start:
            date_filter["$gte"] = filters.date_range.start.isoformat()
        if filters.date_range.end:
            date_filter["$lte"] = filters.date_range.end.isoformat()
        if date_filter:
            pine_filter["date"] = date_filter
    return pine_filter or None


def _compute_confidence(chunks) -> float:
    if not chunks:
        return 0.0
    scores = [c.score if hasattr(c, "score") else c.get("score", 0.0) for c in chunks]
    return min(1.0, float(sum(scores) / len(scores)))


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    if _retrieval_engine is None or _llm_client is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    query_id = str(uuid.uuid4())
    t0 = time.time()

    pinecone_filters = _build_pinecone_filters(request.filters)
    chunks = _retrieval_engine.hybrid_retrieve(
        query=request.question,
        filters=pinecone_filters,
        top_k=request.top_k,
    )

    result = _llm_client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_message=request.question,
        context_chunks=chunks,
    )

    latency_ms = (time.time() - t0) * 1000
    confidence = _compute_confidence(chunks)

    sources = [
        SourceInfo(index=s["index"], source=s["source"], score=s["score"])
        for s in result.sources
    ]

    return QueryResponse(
        query_id=query_id,
        answer=result.answer,
        sources=sources,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 2),
        model=result.model,
    )


@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    logger.info(
        "Feedback received: query_id=%s rating=%d", request.query_id, request.rating
    )
    return {"status": "accepted", "query_id": request.query_id}


@router.get("/history", response_model=List[QueryHistoryItem])
async def history(page: int = 1, page_size: int = 20):
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="page_size must be between 1 and 100")
    # Returns empty list when no DB is wired; callers wire _db_session_factory
    return []


@router.websocket("/stream")
async def stream_query(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        question = data.get("question", "")
        if not question:
            await websocket.send_json({"error": "question is required"})
            await websocket.close()
            return

        if _retrieval_engine is None or _llm_client is None:
            await websocket.send_json({"error": "RAG pipeline not initialized"})
            await websocket.close()
            return

        chunks = _retrieval_engine.retrieve(query=question, top_k=5)
        context = _llm_client.format_context(chunks)

        anthropic_client = anthropic.Anthropic()
        system = (
            f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
        )

        with anthropic_client.messages.stream(
            model=_llm_client.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            for text_chunk in stream.text_stream:
                await websocket.send_json({"token": text_chunk})

        sources = [
            {"index": i + 1, "source": c.source}
            for i, c in enumerate(chunks)
        ]
        await websocket.send_json({"done": True, "sources": sources})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket stream error: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        await websocket.close()


def configure_pipeline(retrieval_engine, llm_client):
    global _retrieval_engine, _llm_client
    _retrieval_engine = retrieval_engine
    _llm_client = llm_client

# _r 20260611131411-048dd078
