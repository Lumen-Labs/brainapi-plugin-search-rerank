from __future__ import annotations

import math
import os
from typing import Any, Callable, Optional

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ESCI_CLASS_GAINS = (1.0, 0.1, 0.01, 0.0)

_predict: Optional[Callable[[list[tuple[str, str]]], list[float]]] = None
_model_name = os.getenv("SEARCH_RERANK_MODEL", DEFAULT_MODEL)
_load_error: Optional[str] = None


def set_predict(fn: Optional[Callable[[list[tuple[str, str]]], list[float]]]) -> None:
    global _predict, _load_error
    _predict = fn
    _load_error = None


def model_name() -> str:
    return _model_name


def status() -> dict[str, Any]:
    return {
        "plugin": "search-rerank",
        "rerank": "plugin:cross-encoder",
        "model": _model_name,
        "loaded": _predict is not None,
        "max_k": 10,
        "error": _load_error,
    }


def _softmax_gain(logits: list[float]) -> float:
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights) or 1.0
    return sum((weight / total) * gain for weight, gain in zip(weights, ESCI_CLASS_GAINS))


def _as_rank_scores(raw: Any) -> list[float]:
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (float, int)):
        return [float(raw)]
    rows = list(raw)
    out: list[float] = []
    for item in rows:
        if isinstance(item, (list, tuple)):
            logits = [float(value) for value in item]
            if len(logits) == len(ESCI_CLASS_GAINS):
                out.append(_softmax_gain(logits))
            elif logits:
                out.append(float(logits[0]))
            else:
                out.append(0.0)
        else:
            out.append(float(item))
    return out


def _ensure_predict() -> Callable[[list[tuple[str, str]]], list[float]]:
    global _predict, _load_error
    if _predict is not None:
        return _predict
    try:
        from sentence_transformers import CrossEncoder

        encoder = CrossEncoder(_model_name)

        def _run(pairs: list[tuple[str, str]]) -> list[float]:
            if not pairs:
                return []
            scores = encoder.predict(pairs)
            return _as_rank_scores(scores)

        _predict = _run
        _load_error = None
        return _predict
    except Exception as exc:
        _load_error = str(exc)
        raise RuntimeError(
            f"Failed to load cross-encoder {_model_name!r}: {exc}"
        ) from exc


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    predict = _ensure_predict()
    pairs = [(query, str(item.get("text") or "")) for item in candidates]
    scores = predict(pairs)
    ranked = []
    for item, score in zip(candidates, scores):
        row = dict(item)
        row["score"] = float(score)
        ranked.append(row)
    ranked.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    limit = max(1, int(k or len(ranked)))
    return ranked[:limit]
