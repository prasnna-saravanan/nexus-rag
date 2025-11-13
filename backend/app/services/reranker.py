"""
Cross-Encoder Reranker.

The "Kill Shot" for high accuracy retrieval:
1. Hybrid Search returns top 20-50 candidates
2. Cross-Encoder scores each (query, document) pair
3. Rerank by cross-encoder scores
4. Return top K

Why it works:
- Bi-encoders (regular embeddings): Query and doc encoded separately
  → Fast but less accurate (can't see query-doc interactions)

- Cross-encoders: Query + doc encoded together
  → Slow but highly accurate (sees full context)

Strategy: Use bi-encoder for fast retrieval (1000s of docs),
          then cross-encoder for precise reranking (top 50)
"""

from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """
    Cross-encoder reranker for precise relevance scoring.

    Models:
    - ms-marco-MiniLM-L-6-v2: Fast, good quality (DEFAULT)
    - ms-marco-electra-base: Slower, higher quality
    - bge-reranker-base: Multilingual support
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lazy_load()

    def _lazy_load(self):
        """Lazy load model (only when first used)."""
        if self.model is None:
            print(f"Loading cross-encoder model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded on {self.device}")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        score_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Search query
            results: List of search results from hybrid search
            top_k: Number of top results to return
            score_key: Key in result dict containing text to score

        Returns:
            Reranked results with cross_encoder_score added
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = []
        for result in results:
            doc_text = result.get(score_key, "")
            pairs.append([query, doc_text])

        # Batch scoring for efficiency
        scores = self._score_batch(pairs)

        # Add cross-encoder scores to results
        for result, score in zip(results, scores):
            result["cross_encoder_score"] = float(score)

        # Sort by cross-encoder score
        reranked = sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)

        return reranked[:top_k]

    def _score_batch(self, pairs: List[List[str]], batch_size: int = 32) -> List[float]:
        """
        Score query-document pairs in batches.

        Args:
            pairs: List of [query, document] pairs
            batch_size: Batch size for processing

        Returns:
            List of relevance scores
        """
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                # Get scores
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Convert to scores (sigmoid for binary classification)
                scores = torch.sigmoid(logits).squeeze(-1)
                all_scores.extend(scores.cpu().tolist())

        return all_scores

    def get_info(self) -> Dict[str, Any]:
        """Return information about the reranker."""
        return {
            "name": "Cross-Encoder Reranker",
            "model": self.model_name,
            "device": self.device,
            "description": "Precise relevance scoring using cross-attention",
            "use_case": "Rerank top candidates from hybrid search",
        }


class DummyReranker:
    """
    Dummy reranker that returns results as-is.
    Use when cross-encoder is not needed or too slow.
    """

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        score_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """Return results unchanged."""
        return results[:top_k]

    def get_info(self) -> Dict[str, Any]:
        """Return info."""
        return {"name": "Dummy Reranker", "description": "No reranking applied"}
