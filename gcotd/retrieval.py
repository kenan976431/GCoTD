"""
GCoTD Template Retrieval
========================

Two-stage process:
  1. Task-class matching   – c = argmax_{c'∈C} Alignment(R, c')
  2. Top-N template select – T_i = argmax_{T∈c} Sim(R, Q_T)

Both stages use sentence-level cosine similarity.  If no template exceeds
the threshold θ, a general fallback template is returned.
"""

import re
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .templates import (
    GCoTTemplate,
    TEMPLATE_REGISTRY,
    TASK_CLASSES,
    get_all_templates,
    format_templates_for_prompt,
)


# ---------------------------------------------------------------------------
# Keyword heuristics for fast task-class alignment
# (augments embedding similarity; helps weak / small embedding models)
# ---------------------------------------------------------------------------

TASK_KEYWORDS: dict[str, list[str]] = {
    "letter": [
        "last letter", "last letters", "concatenate", "letters of",
        "first letter", "spell", "acronym",
    ],
    "math": [
        "calculate", "compute", "solve", "equation", "coefficient",
        "geometric", "arithmetic", "algebra", "probability", "proof",
        "how many", "total", "sum", "product", "integral", "derivative",
        "matrix", "polynomial",
    ],
    "csqa": [
        "answer choices", "which of the following", "what is", "where would",
        "(a)", "(b)", "(c)", "(d)", "(e)", "multiple choice",
        "commonsense", "common sense",
    ],
    "general": [],  # catch-all
}


# ---------------------------------------------------------------------------
# Embedding utility
# ---------------------------------------------------------------------------

def _get_embedder():
    """
    Lazily load a sentence-transformer encoder.
    Falls back to a simple TF-IDF cosine if sentence-transformers is absent.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def embed(texts: list[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True)

        return embed
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        all_texts = [t.question for t in get_all_templates()]
        vectorizer.fit(all_texts)

        def embed(texts: list[str]) -> np.ndarray:  # type: ignore[misc]
            mat = vectorizer.transform(texts).toarray()
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            return mat / norms

        return embed


_EMBEDDER = None


def _embed(texts: list[str]) -> np.ndarray:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _get_embedder()
    return _EMBEDDER(texts)


# ---------------------------------------------------------------------------
# Stage 1: Task-class alignment
# ---------------------------------------------------------------------------

def _keyword_score(text: str, task: str) -> float:
    """Keyword overlap score for fast pre-filtering."""
    text_lower = text.lower()
    keywords = TASK_KEYWORDS.get(task, [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def classify_task(response: str, use_embedding: bool = True) -> str:
    """
    c = argmax_{c'∈C} Alignment(R, c')

    Combines keyword heuristics (fast) with embedding similarity (accurate).
    Returns the best-matching task class string from TASK_CLASSES.
    """
    # --- keyword pass ---
    kw_scores: dict[str, float] = {
        task: _keyword_score(response, task) for task in TASK_CLASSES
        if task != "general"
    }
    best_kw = max(kw_scores, key=kw_scores.get) if kw_scores else "general"
    best_kw_score = kw_scores.get(best_kw, 0.0)

    # Strong keyword signal → use it directly
    if best_kw_score >= 0.15:
        return best_kw

    if not use_embedding:
        return best_kw if best_kw_score > 0 else "general"

    # --- embedding pass over representative questions per class ---
    class_representatives: dict[str, list[str]] = {
        task: [t.question for t in templates]
        for task, templates in TEMPLATE_REGISTRY.items()
        if task != "general"
    }

    response_emb = _embed([response])
    class_scores: dict[str, float] = {}
    for task, questions in class_representatives.items():
        q_embs = _embed(questions)
        sims = cosine_similarity(response_emb, q_embs)[0]
        class_scores[task] = float(sims.max())

    # blend: 0.4 × keyword + 0.6 × embedding
    blended: dict[str, float] = {}
    for task in class_scores:
        blended[task] = (0.4 * kw_scores.get(task, 0.0) +
                         0.6 * class_scores[task])

    best_task = max(blended, key=blended.get)
    return best_task if blended[best_task] > 0.1 else "general"


# ---------------------------------------------------------------------------
# Stage 2: Top-N template retrieval  (Eq. 2)
# ---------------------------------------------------------------------------

def retrieve_templates(
    response: str,
    task_class: Optional[str] = None,
    top_n: int = 3,
    similarity_threshold: float = 0.3,
) -> list[GCoTTemplate]:
    """
    Eq. (2): T_i = argmax_{T∈c} Sim(R, Q_T)

    Retrieves the top-N most similar GCoT templates from the task class.
    If Sim(R, Q_T) < θ for all templates, falls back to general templates.

    Args:
        response            : the LLM response text R
        task_class          : pre-determined class c (auto-detected if None)
        top_n               : number of templates N to return
        similarity_threshold: θ – minimum cosine similarity to be included

    Returns:
        List of up to top_n GCoTTemplate objects, sorted by similarity desc.
    """
    if task_class is None:
        task_class = classify_task(response)

    candidate_pool = TEMPLATE_REGISTRY.get(task_class, [])

    # Fallback: include general templates if pool is small
    if len(candidate_pool) < top_n:
        candidate_pool = candidate_pool + TEMPLATE_REGISTRY.get("general", [])

    if not candidate_pool:
        return []

    questions = [t.question for t in candidate_pool]
    response_emb = _embed([response])
    q_embs = _embed(questions)
    sims = cosine_similarity(response_emb, q_embs)[0]   # (N_pool,)

    # Sort by similarity descending
    ranked_idx = np.argsort(sims)[::-1]
    selected: list[GCoTTemplate] = []
    for idx in ranked_idx:
        if sims[idx] < similarity_threshold:
            break
        selected.append(candidate_pool[idx])
        if len(selected) >= top_n:
            break

    if not selected:
        selected = TEMPLATE_REGISTRY.get("general", candidate_pool[:top_n])

    return selected[:top_n]


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

class TemplateRetriever:
    """
    Stateful wrapper combining task classification and template retrieval.
    """

    def __init__(
        self,
        top_n: int = 3,
        similarity_threshold: float = 0.3,
        use_embedding: bool = True,
    ):
        self.top_n = top_n
        self.similarity_threshold = similarity_threshold
        self.use_embedding = use_embedding

    def retrieve(self, response: str) -> tuple[str, list[GCoTTemplate], str]:
        """
        Full retrieval pipeline for a given LLM response R.

        Returns:
            (task_class, templates, formatted_prompt_block)
        """
        task_class = classify_task(response, use_embedding=self.use_embedding)
        templates = retrieve_templates(
            response,
            task_class=task_class,
            top_n=self.top_n,
            similarity_threshold=self.similarity_threshold,
        )
        prompt_block = format_templates_for_prompt(templates)
        return task_class, templates, prompt_block

    def __repr__(self) -> str:
        return (f"TemplateRetriever(top_n={self.top_n}, "
                f"threshold={self.similarity_threshold})")
