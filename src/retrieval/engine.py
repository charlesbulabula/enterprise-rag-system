import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    sparse_results: List[RetrievedChunk],
    k: int = 60,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    scores: Dict[str, float] = {}
    chunks_by_id: Dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        uid = chunk.chunk_id or chunk.content[:64]
        scores[uid] = scores.get(uid, 0.0) + _rrf_score(rank, k)
        chunks_by_id[uid] = chunk

    for rank, chunk in enumerate(sparse_results):
        uid = chunk.chunk_id or chunk.content[:64]
        scores[uid] = scores.get(uid, 0.0) + _rrf_score(rank, k)
        if uid not in chunks_by_id:
            chunks_by_id[uid] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused: List[RetrievedChunk] = []
    for uid in sorted_ids[:top_k]:
        chunk = chunks_by_id[uid]
        fused.append(
            RetrievedChunk(
                content=chunk.content,
                score=scores[uid],
                source=chunk.source,
                metadata=chunk.metadata,
                chunk_id=chunk.chunk_id,
            )
        )
    return fused


class RetrievalEngine:
    def __init__(
        self,
        pinecone_index,
        embedding_service,
        top_k: int = 5,
        bm25_corpus: Optional[List[Dict[str, Any]]] = None,
    ):
        self.pinecone_index = pinecone_index
        self.embedding_service = embedding_service
        self.top_k = top_k
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict[str, Any]] = []

        if bm25_corpus:
            self._build_bm25_index(bm25_corpus)

    def _build_bm25_index(self, corpus: List[Dict[str, Any]]) -> None:
        self._bm25_docs = corpus
        tokenized = [doc["content"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 index with %d documents", len(corpus))

    def _pinecone_to_chunks(self, matches: List[Dict]) -> List[RetrievedChunk]:
        chunks = []
        for match in matches:
            metadata = match.get("metadata", {})
            chunks.append(
                RetrievedChunk(
                    content=metadata.get("text", metadata.get("content", "")),
                    score=float(match.get("score", 0.0)),
                    source=metadata.get("source", "unknown"),
                    metadata=metadata,
                    chunk_id=match.get("id"),
                )
            )
        return chunks

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or self.top_k
        query_vector = self.embedding_service.embed(query)

        query_kwargs: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": k,
            "include_metadata": True,
        }
        if filters:
            query_kwargs["filter"] = filters

        response = self.pinecone_index.query(**query_kwargs)
        matches = response.get("matches", [])
        chunks = self._pinecone_to_chunks(matches)

        logger.debug("Dense retrieval returned %d chunks for query: %r", len(chunks), query[:80])
        return chunks

    def hybrid_retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or self.top_k

        dense_results = self.retrieve(query, filters=filters, top_k=k * 2)

        sparse_results: List[RetrievedChunk] = []
        if self._bm25 is not None and self._bm25_docs:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[: k * 2]

            for idx in top_indices:
                doc = self._bm25_docs[idx]
                metadata = doc.get("metadata", {})
                sparse_results.append(
                    RetrievedChunk(
                        content=doc["content"],
                        score=float(bm25_scores[idx]),
                        source=metadata.get("source", "unknown"),
                        metadata=metadata,
                        chunk_id=doc.get("id"),
                    )
                )
        else:
            logger.warning("BM25 index not available; falling back to dense-only retrieval")
            return dense_results[:k]

        fused = _reciprocal_rank_fusion(dense_results, sparse_results, top_k=k)
        logger.debug(
            "Hybrid retrieval: dense=%d sparse=%d fused=%d",
            len(dense_results),
            len(sparse_results),
            len(fused),
        )
        return fused

    def update_bm25_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        self._build_bm25_index(corpus)

# _r 20260518204217-7b244bdc
