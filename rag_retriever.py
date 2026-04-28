"""
RAG Retriever for PawPal+
Loads a structured JSON knowledge base and retrieves the most relevant passages
for a user query using TF-IDF cosine similarity via scikit-learn.
"""

import json
import os
from typing import List, Dict, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RAGRetriever:
    """
    Retrieval-Augmented Generation knowledge base using TF-IDF.

    Loads all .json files from the knowledge_base directory, indexes every
    entry's title + content + tags, and returns the top-k most similar
    passages for any query.
    """

    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base_dir = knowledge_base_dir
        self.documents: List[Dict] = []
        self._tfidf_matrix = None
        self._vectorizer: Optional[object] = None
        self._available = SKLEARN_AVAILABLE

        if not self._available:
            return

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            ngram_range=(1, 2),   # unigrams + bigrams for better phrase matching
            sublinear_tf=True,    # apply log normalization to term frequency
        )

        self._load_knowledge_base()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_knowledge_base(self) -> None:
        """Parse every .json file in the knowledge_base directory."""
        if not os.path.isdir(self.knowledge_base_dir):
            return

        for filename in sorted(os.listdir(self.knowledge_base_dir)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.knowledge_base_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            category = data.get("metadata", {}).get("category", "general")
            source_label = (
                filename.replace(".json", "").replace("_", " ").title()
            )

            for entry in data.get("entries", []):
                # Build a rich text representation for indexing
                parts = [
                    entry.get("title", ""),
                    entry.get("content", ""),
                    " ".join(entry.get("tags", [])),
                    " ".join(entry.get("species", [])),
                    entry.get("when_to_see_vet", ""),
                ]
                index_text = " ".join(p for p in parts if p)

                self.documents.append(
                    {
                        "id": entry.get("id", ""),
                        "title": entry.get("title", ""),
                        "content": entry.get("content", ""),
                        "category": category,
                        "source": source_label,
                        "filename": filename,
                        "tags": entry.get("tags", []),
                        "species": entry.get("species", ["all"]),
                        "urgency": entry.get("urgency", "low"),
                        "when_to_see_vet": entry.get("when_to_see_vet", ""),
                        "_index_text": index_text,
                    }
                )

        if self.documents:
            texts = [doc["_index_text"] for doc in self.documents]
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """
        Return the top-k most relevant documents for *query*.

        Each returned dict has the original document fields plus a float
        'score' key (cosine similarity, 0-1).  Returns [] when scikit-learn
        is unavailable or the knowledge base is empty.
        """
        if not self._available or not self.documents or self._tfidf_matrix is None:
            return []

        try:
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        except Exception:
            return []

        # Rank by score descending, take top-k with non-zero similarity
        ranked_indices = scores.argsort()[::-1]
        results = []
        for idx in ranked_indices:
            if len(results) >= k:
                break
            score = float(scores[idx])
            if score <= 0.0:
                break
            doc = {k: v for k, v in self.documents[idx].items() if k != "_index_text"}
            doc["score"] = round(score, 4)
            results.append(doc)

        return results

    def retrieve_by_species(self, query: str, species: str, k: int = 4) -> List[Dict]:
        """Retrieve top-k docs filtered to entries matching *species*."""
        all_results = self.retrieve(query, k=k * 3)
        filtered = [
            d for d in all_results
            if "all" in d.get("species", []) or species.lower() in [s.lower() for s in d.get("species", [])]
        ]
        return filtered[:k]

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_document_count(self) -> int:
        """Total number of indexed entries."""
        return len(self.documents)

    def get_categories(self) -> List[str]:
        """Unique category names in the knowledge base."""
        return sorted(set(d["category"] for d in self.documents))

    def is_available(self) -> bool:
        """True when scikit-learn is installed and the KB loaded successfully."""
        return self._available and len(self.documents) > 0

    def get_status(self) -> Dict:
        """Summary dict for the system reports tab."""
        return {
            "available": self.is_available(),
            "document_count": self.get_document_count(),
            "categories": self.get_categories(),
            "sklearn_installed": SKLEARN_AVAILABLE,
        }
