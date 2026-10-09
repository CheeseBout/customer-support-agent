"""Calibrate the retrieval relevance gate (`retrieval.score_threshold`) on a shop's own documents.

The gate rejects a question when its best policy chunk scores below the threshold. The right
value depends on the embedding model and on the documents, so a number tuned on one shop's
documents is wrong for another. This module needs only two lists of questions: ones the
documents answer, and ones they do not. It calls no chat model.

Questions file, YAML or JSONL (one object per line):

    - q: How many days do I have to return an item?
      answerable: true
    - q: What is the weather like today?
      answerable: false
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_agent.evals import metrics as m
from support_agent.evals.runner import GateReport
from support_agent.rag.retriever import Retriever

MIN_ANSWERABLE = 10
MIN_UNANSWERABLE = 5
STEP = 0.005


class CalibrationError(ValueError):
    """The questions file is unusable."""


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=2)
    answerable: bool
    id: str | None = None


class Calibration(BaseModel):
    gate: GateReport
    n_questions: int
    warnings: list[str] = Field(default_factory=list)

    @property
    def recommended(self) -> float | None:
        return self.gate.recommended_threshold

    @property
    def separable(self) -> bool:
        """True when every answerable question scores above every unanswerable one."""
        g = self.gate
        return (
            g.lowest_positive is not None
            and g.highest_negative is not None
            and g.highest_negative < g.lowest_positive
        )


def load_questions(path: Path) -> list[Question]:
    if not path.exists():
        raise CalibrationError(f"questions file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            raw = yaml.safe_load(text) or []
            if isinstance(raw, dict) and "questions" in raw:
                raw = raw["questions"]
        else:
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(raw, list):
            raise CalibrationError(f"{path}: expected a list of questions")
        questions = [Question.model_validate(item) for item in raw]
    except (ValidationError, ValueError, yaml.YAMLError) as exc:
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError(f"{path}: {exc}") from exc
    return [
        q if q.id else q.model_copy(update={"id": f"q{i:03d}"})
        for i, q in enumerate(questions, start=1)
    ]


def _thresholds(scores: list[float]) -> list[float]:
    """Candidate thresholds spanning what was observed: models score in very different ranges."""
    lo = max(0.0, math.floor((min(scores) - 0.05) / STEP) * STEP)
    hi = min(1.0, math.ceil((max(scores) + 0.05) / STEP) * STEP)
    count = int(round((hi - lo) / STEP))
    return [round(lo + i * STEP, 3) for i in range(count + 1)]


def calibrate(
    retriever: Retriever, questions: list[Question], *, configured_threshold: float
) -> Calibration:
    """Best chunk score per question (no gate applied), then the threshold that separates them."""
    positives: dict[str, float] = {}
    negatives: dict[str, float] = {}
    for question in questions:
        hits = retriever.retrieve_sync(question.q, top_k=retriever.cfg.top_k, threshold=0.0)
        best = max((h.score for h in hits), default=0.0)
        (positives if question.answerable else negatives)[question.id or question.q] = best

    warnings: list[str] = []
    if len(positives) < MIN_ANSWERABLE:
        warnings.append(
            f"Only {len(positives)} answerable questions (at least {MIN_ANSWERABLE} recommended): "
            "the result is a rough guide."
        )
    if len(negatives) < MIN_UNANSWERABLE:
        warnings.append(
            f"Only {len(negatives)} unanswerable questions (at least {MIN_UNANSWERABLE} "
            "recommended): a threshold cannot be checked against what it must reject."
        )

    pos, neg = list(positives.values()), list(negatives.values())
    everything = pos + neg
    if not everything:
        raise CalibrationError("no questions to calibrate on")
    curve = m.gate_curve(pos, neg, _thresholds(everything))
    at_config = m.gate_curve(pos, neg, [configured_threshold])[0]
    gate = GateReport(
        configured_threshold=configured_threshold,
        recall=at_config.recall if pos else None,
        specificity=at_config.specificity if neg else None,
        recommended_threshold=m.recommend_threshold(pos, neg, curve),
        n_positive=len(pos),
        n_negative=len(neg),
        lowest_positive=min(pos) if pos else None,
        highest_negative=max(neg) if neg else None,
        lowest_positives=sorted(positives.items(), key=lambda kv: kv[1])[:5],
        highest_negatives=sorted(negatives.items(), key=lambda kv: -kv[1])[:5],
        curve=[
            {"threshold": p.threshold, "recall": p.recall, "specificity": p.specificity}
            for p in curve
        ],
    )
    result = Calibration(gate=gate, n_questions=len(questions), warnings=warnings)
    if pos and neg and not result.separable:
        result.warnings.append(
            "Some unanswerable questions score as high as answerable ones, so no threshold "
            "separates them. The recommendation balances the two; the chat model's own "
            "judgement of whether the text answers the question catches the rest. A false "
            "rejection costs more than a borderline pass, so prefer the lower value when unsure."
        )
    return result
