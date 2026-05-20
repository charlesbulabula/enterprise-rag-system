import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 8000
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


@dataclass
class GenerationResult:
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int
    model: str
    latency_ms: float = 0.0


class LLMClient:
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def format_context(self, chunks) -> str:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.source if hasattr(chunk, "source") else chunk.get("source", "unknown")
            content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
            lines.append(f"[Source {i}: {source}]\n{content}")
        return "\n\n---\n\n".join(lines)

    def _build_sources_list(self, chunks) -> List[Dict[str, Any]]:
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.source if hasattr(chunk, "source") else chunk.get("source", "unknown")
            metadata = chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {})
            score = chunk.score if hasattr(chunk, "score") else chunk.get("score", 0.0)
            sources.append(
                {
                    "index": i,
                    "source": source,
                    "score": score,
                    "metadata": metadata,
                }
            )
        return sources

    def _call_anthropic(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                return {
                    "answer": response.content[0].text,
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Anthropic API error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

    def _call_openai(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                return {
                    "answer": response.choices[0].message.content,
                    "tokens_used": response.usage.total_tokens,
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "OpenAI API error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)
                else:
                    raise

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: List[Any],
    ) -> GenerationResult:
        context_str = self.format_context(context_chunks)
        sources = self._build_sources_list(context_chunks)

        augmented_system = (
            f"{system_prompt}\n\n"
            "Use ONLY the following context to answer the question. "
            "Cite sources using [Source N] notation where N corresponds to the source index. "
            "If the answer cannot be found in the context, say so explicitly.\n\n"
            f"CONTEXT:\n{context_str}"
        )

        t0 = time.time()
        if self.provider == "anthropic":
            result = self._call_anthropic(augmented_system, user_message)
        else:
            result = self._call_openai(augmented_system, user_message)

        latency_ms = (time.time() - t0) * 1000

        logger.info(
            "Generated response: provider=%s model=%s tokens=%d latency=%.1fms",
            self.provider,
            self.model,
            result["tokens_used"],
            latency_ms,
        )

        return GenerationResult(
            answer=result["answer"],
            sources=sources,
            tokens_used=result["tokens_used"],
            model=self.model,
            latency_ms=round(latency_ms, 2),
        )

# _r 20260520152605-7e1e4b7f
