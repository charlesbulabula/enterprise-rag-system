import hashlib
import json
import logging
import time
from typing import List, Optional

import redis

logger = logging.getLogger(__name__)

_OPENAI_MAX_BATCH = 2048
_VOYAGE_MAX_BATCH = 128


class EmbeddingService:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        redis_url: Optional[str] = "redis://localhost:6379",
        cache_ttl: int = 86400,
        max_retries: int = 5,
    ):
        self.provider = provider
        self.model = model
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries

        if provider == "openai":
            import openai
            self.client = openai.OpenAI()
        elif provider == "anthropic":
            import voyageai
            self.client = voyageai.Client()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        self._redis: Optional[redis.Redis] = None
        if redis_url:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Redis cache connected at %s", redis_url)
            except Exception as e:
                logger.warning("Redis unavailable, caching disabled: %s", e)
                self._redis = None

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb:{self.provider}:{self.model}:{digest}"

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        if self._redis is None:
            return None
        try:
            key = self._cache_key(text)
            cached = self._redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug("Cache read error: %s", e)
        return None

    def _set_in_cache(self, text: str, embedding: List[float]) -> None:
        if self._redis is None:
            return
        try:
            key = self._cache_key(text)
            self._redis.setex(key, self.cache_ttl, json.dumps(embedding))
        except Exception as e:
            logger.debug("Cache write error: %s", e)

    def _call_with_backoff(self, fn, *args, **kwargs):
        delay = 1.0
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Embedding API error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
        raise last_exc

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        response = self._call_with_backoff(
            self.client.embeddings.create,
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]

    def _embed_voyage(self, texts: List[str]) -> List[List[float]]:
        result = self._call_with_backoff(
            self.client.embed,
            texts,
            model=self.model,
            input_type="document",
        )
        return result.embeddings

    def embed(self, text: str) -> List[float]:
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        if self.provider == "openai":
            embeddings = self._embed_openai([text])
        else:
            embeddings = self._embed_voyage([text])

        result = embeddings[0]
        self._set_in_cache(text, result)
        return result

    def embed_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            max_batch = _OPENAI_MAX_BATCH if self.provider == "openai" else _VOYAGE_MAX_BATCH
            effective_batch = min(batch_size, max_batch)

            for batch_start in range(0, len(uncached_texts), effective_batch):
                batch = uncached_texts[batch_start: batch_start + effective_batch]
                if self.provider == "openai":
                    batch_embeddings = self._embed_openai(batch)
                else:
                    batch_embeddings = self._embed_voyage(batch)

                for local_idx, embedding in enumerate(batch_embeddings):
                    global_idx = uncached_indices[batch_start + local_idx]
                    results[global_idx] = embedding
                    self._set_in_cache(uncached_texts[batch_start + local_idx], embedding)

        logger.info(
            "Embedded %d texts (%d from cache, %d fresh)",
            len(texts),
            len(texts) - len(uncached_texts),
            len(uncached_texts),
        )
        return results

# _r 20260611100206-f9d9f667
