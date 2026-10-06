"""Check if generated feedback is supported by retrieved context."""

import math
import re

import structlog

from core.config import settings
from ingestion.embeddings.provider import EmbeddingProvider, get_embedding_provider

logger = structlog.get_logger()

# Short claims with little word overlap are compared by meaning instead
SHORT_CLAIM_MAX_WORDS = 4
SEMANTIC_SUPPORT_THRESHOLD = 0.5


def _semantic_score(claim: str, context: str, provider: EmbeddingProvider | None = None) -> float:
    """Cosine similarity between the embedded claim and the embedded context.

    Args:
        claim: Claim text
        context: Context text
        provider: Embedding provider; defaults to the one named in settings

    Returns:
        Cosine similarity of the two embeddings
    """
    if provider is None:
        provider = get_embedding_provider(settings.llm_provider)

    claim_vec, context_vec = provider.embed([claim, context])
    dot = sum(a * b for a, b in zip(claim_vec, context_vec))
    norm = math.sqrt(sum(a * a for a in claim_vec)) * math.sqrt(sum(b * b for b in context_vec))
    return dot / norm if norm else 0.0


class FaithfulnessChecker:
    """Verify that feedback claims are supported by context."""

    def check(self, feedback: str, context_chunks: list[dict]) -> float:
        """Check faithfulness of feedback to context.

        Args:
            feedback: Generated feedback text
            context_chunks: Retrieved context chunks

        Returns:
            Faithfulness score 0.0-1.0 (ratio of supported claims)
        """
        if not feedback or not context_chunks:
            logger.info(
                "faithfulness_empty_input",
                has_feedback=bool(feedback),
                has_chunks=bool(context_chunks),
            )
            return 0.0

        # Extract key claims from feedback (sentences)
        claims = self._extract_claims(feedback)
        if not claims:
            logger.info("faithfulness_no_claims_extracted")
            return 0.5  # Default to neutral if no extractable claims

        # Concatenate context text
        context_text = " ".join([chunk.get("text", "") for chunk in context_chunks])

        # Check each claim for support
        supported = 0
        for claim in claims:
            if self._is_supported(claim, context_text):
                supported += 1

        score = supported / len(claims) if claims else 0.0

        logger.info(
            "faithfulness_checked", claims_count=len(claims), supported_count=supported, score=score
        )

        return score

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        """Extract key claims from feedback text.

        Args:
            text: Feedback text

        Returns:
            List of claims (sentences)
        """
        # Split by sentence (simple regex)
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return claims[:10]  # Limit to 10 claims for scoring

    @staticmethod
    def _is_supported(claim: str, context: str, provider: EmbeddingProvider | None = None) -> bool:
        """Check if a claim is supported by context.

        Args:
            claim: Claim text
            context: Context text
            provider: Embedding provider for the semantic fallback (defaults to settings)

        Returns:
            True if claim is supported
        """
        # Tokenize and check for keyword overlap
        claim_tokens = set(claim.lower().split())
        context_tokens = set(context.lower().split())

        # Require at least some meaningful overlap
        overlap = claim_tokens & context_tokens
        # Filter out common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "and",
            "or",
            "but",
            "in",
            "of",
            "to",
            "for",
            "that",
        }
        meaningful_overlap = overlap - stop_words

        if len(meaningful_overlap) >= 2:
            return True

        # Short claims rarely share two words with the context even when the meaning
        # matches, so fall back to comparing embeddings
        if len(claim.split()) <= SHORT_CLAIM_MAX_WORDS:
            try:
                return _semantic_score(claim, context, provider) > SEMANTIC_SUPPORT_THRESHOLD
            except Exception:
                logger.warning("faithfulness_semantic_fallback_failed", exc_info=True)

        return False
