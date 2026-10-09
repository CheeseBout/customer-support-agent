"""`support-agent calibrate`: find the relevance threshold for a shop's own documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from support_agent.evals.calibrate import (
    Calibration,
    CalibrationError,
    Question,
    calibrate,
    load_questions,
)
from tests.conftest import DEMO


@dataclass
class Hit:
    score: float


class ScriptedRetriever:
    """Returns a fixed best score per query, like an index would."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.cfg = type("Cfg", (), {"top_k": 6})()

    def retrieve_sync(self, query: str, *, top_k: int, threshold: float) -> list[Hit]:
        assert threshold == 0.0  # calibration looks at raw scores, never at a gate
        return [Hit(self.scores[query])]


def questions(answerable: list[float], unanswerable: list[float]) -> tuple[list[Question], dict]:
    items, scores = [], {}
    for kind, values in ((True, answerable), (False, unanswerable)):
        for i, score in enumerate(values):
            q = f"{'yes' if kind else 'no'} {i}"
            items.append(Question(q=q, answerable=kind))
            scores[q] = score
    return items, scores


def run(answerable: list[float], unanswerable: list[float], threshold: float = 0.8) -> Calibration:
    items, scores = questions(answerable, unanswerable)
    return calibrate(ScriptedRetriever(scores), items, configured_threshold=threshold)


# --- loading ---------------------------------------------------------------------------------


def test_yaml_and_jsonl_questions_load_with_generated_ids(tmp_path: Path):
    (tmp_path / "q.yaml").write_text(
        '- {q: "How do returns work?", answerable: true}\n- {q: "Weather today?", answerable: false}\n',
        encoding="utf-8",
    )
    (tmp_path / "q.jsonl").write_text(
        '{"q": "How do returns work?", "answerable": true}\n\n'
        '{"q": "Weather today?", "answerable": false, "id": "w"}\n',
        encoding="utf-8",
    )
    from_yaml, from_jsonl = (
        load_questions(tmp_path / "q.yaml"),
        load_questions(tmp_path / "q.jsonl"),
    )
    assert [q.id for q in from_yaml] == ["q001", "q002"]
    assert [q.id for q in from_jsonl] == ["q001", "w"]
    assert [q.answerable for q in from_yaml] == [True, False]


@pytest.mark.parametrize(
    "text",
    [
        "- {q: only a question}\n",
        "- {q: x, answerable: true}\n",
        "not: a list\n",
        "- {q: ok, answerable: maybe}\n",
        "- {q: Does it work?, answerable: true}\n",  # an unquoted "?" is not valid YAML here
    ],
)
def test_malformed_questions_are_rejected(tmp_path: Path, text: str):
    path = tmp_path / "q.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_questions(path)


def test_a_missing_file_is_reported(tmp_path: Path):
    with pytest.raises(CalibrationError, match="not found"):
        load_questions(tmp_path / "nope.yaml")


def test_the_demo_questions_are_valid():
    items = load_questions(DEMO / "evals" / "calibration.yaml")
    assert sum(q.answerable for q in items) >= 30 and sum(not q.answerable for q in items) >= 10


# --- the recommendation -----------------------------------------------------------------------------------


def test_clean_separation_recommends_the_middle_of_the_gap():
    result = run([0.86, 0.88, 0.90] * 4, [0.71, 0.74, 0.76] * 2)
    assert result.separable and result.recommended == pytest.approx(0.81, abs=0.001)
    assert result.gate.recall == 1.0 and result.gate.specificity == 1.0
    assert result.warnings == []


def test_it_works_in_any_score_range_not_only_that_of_the_default_model():
    # text-embedding-3-small style scores sit far below e5 scores.
    result = run([0.42, 0.45, 0.50] * 4, [0.15, 0.20, 0.25] * 2, threshold=0.8)
    assert result.recommended == pytest.approx(0.335, abs=0.001)
    assert result.gate.recall == 0.0  # the configured 0.80 would reject every real question
    assert result.gate.curve[0]["threshold"] < 0.2


def test_overlapping_scores_warn_and_favour_a_balanced_threshold():
    result = run([0.70, 0.80, 0.85, 0.90] * 3, [0.60, 0.82, 0.65] * 2)
    assert not result.separable
    assert any("no threshold separates" in w for w in result.warnings)
    assert result.recommended is not None


def test_few_questions_are_flagged():
    result = run([0.9, 0.9], [0.5])
    assert any("answerable" in w for w in result.warnings)
    assert any("unanswerable" in w for w in result.warnings)


def test_one_class_only_gives_no_recommendation():
    result = run([0.9] * 12, [])
    assert result.recommended is None and result.gate.n_negative == 0


def test_no_questions_is_an_error():
    with pytest.raises(CalibrationError, match="no questions"):
        calibrate(ScriptedRetriever({}), [], configured_threshold=0.8)
