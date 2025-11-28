"""Custom LLM-as-a-Judge evaluator for RAG quality assessment.

This module provides a lightweight LLM-based evaluation system that doesn't
require the full Ragas framework. It's useful for quick validation and
custom evaluation criteria.

Environment variables for threshold configuration:
    LLM_JUDGE_FAITHFULNESS_THRESHOLD: Minimum faithfulness score (default: 0.8)
    LLM_JUDGE_RELEVANCY_THRESHOLD: Minimum relevancy score (default: 0.7)
"""

import json
import logging
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden via environment variables)
DEFAULT_FAITHFULNESS_THRESHOLD = 0.8
DEFAULT_RELEVANCY_THRESHOLD = 0.7


def _get_threshold(env_var: str, default: float) -> float:
    """Get threshold from environment variable or use default."""
    value = os.environ.get(env_var)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            logger.warning(
                f"Invalid threshold value for {env_var}: {value}, using default {default}"
            )
    return default


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation.

    Attributes:
        score: Quality score between 0.0 and 1.0.
        reason: Human-readable explanation for the score.
        passed: Whether the score meets the threshold.
        details: Additional evaluation details.
    """

    score: float
    reason: str
    passed: bool
    details: dict | None = None


class LLMJudge:
    """Custom LLM-as-a-Judge evaluator.

    Uses a smaller, faster model (GPT-4o-mini) to evaluate the quality
    of RAG-generated answers. Provides custom evaluation metrics beyond
    what Ragas offers.

    Example:
        >>> judge = LLMJudge(client)
        >>> result = await judge.evaluate_faithfulness(
        ...     question="What is Python?",
        ...     answer="Python is a programming language.",
        ...     context=["Python is a high-level programming language."],
        ... )
        >>> print(f"Score: {result.score}, Passed: {result.passed}")
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        faithfulness_threshold: float | None = None,
        relevancy_threshold: float | None = None,
    ) -> None:
        """Initialize the LLM Judge.

        Args:
            client: AsyncOpenAI client for API calls.
            model: Model to use for evaluation (default: gpt-4o-mini for cost).
            faithfulness_threshold: Minimum score for faithfulness to pass.
                If None, reads from LLM_JUDGE_FAITHFULNESS_THRESHOLD env var
                or uses default (0.8).
            relevancy_threshold: Minimum score for relevancy to pass.
                If None, reads from LLM_JUDGE_RELEVANCY_THRESHOLD env var
                or uses default (0.7).
        """
        self._client = client
        self._model = model
        self._faithfulness_threshold = (
            faithfulness_threshold
            if faithfulness_threshold is not None
            else _get_threshold(
                "LLM_JUDGE_FAITHFULNESS_THRESHOLD", DEFAULT_FAITHFULNESS_THRESHOLD
            )
        )
        self._relevancy_threshold = (
            relevancy_threshold
            if relevancy_threshold is not None
            else _get_threshold(
                "LLM_JUDGE_RELEVANCY_THRESHOLD", DEFAULT_RELEVANCY_THRESHOLD
            )
        )

    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> JudgeResult:
        """Evaluate if the answer is faithful to the context.

        Faithfulness measures whether all claims in the answer can be
        verified from the provided context. A high score indicates
        low hallucination risk.

        Args:
            question: The original question asked.
            answer: The generated answer to evaluate.
            context: List of context passages used for generation.

        Returns:
            JudgeResult with faithfulness score and explanation.
        """
        context_text = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))

        prompt = f"""You are an impartial judge evaluating the faithfulness of an AI assistant's answer.

**Task:** Determine if the answer is faithful (grounded) in the provided context.

**Question:** {question}

**Context provided to the assistant:**
{context_text}

**Assistant's Answer:** {answer}

**Evaluation Criteria:**
- Score 1.0: All claims in the answer are directly supported by the context
- Score 0.8-0.9: Most claims are supported, minor inferences are reasonable
- Score 0.5-0.7: Some claims are supported, but significant additions exist
- Score 0.2-0.4: Few claims are supported, mostly unsupported statements
- Score 0.0-0.1: Answer contradicts or ignores the context entirely

**Instructions:**
1. Identify each factual claim in the answer
2. Check if each claim can be verified from the context
3. Penalize claims that cannot be verified (hallucinations)
4. Consider if the answer appropriately indicates uncertainty when context is insufficient

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation of the score>",
    "verified_claims": ["<list of claims supported by context>"],
    "unverified_claims": ["<list of claims not supported by context>"]
}}"""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            result = json.loads(response.choices[0].message.content or "{}")
            score = float(result.get("score", 0.0))

            return JudgeResult(
                score=score,
                reason=result.get("reason", "No reason provided"),
                passed=score >= self._faithfulness_threshold,
                details={
                    "verified_claims": result.get("verified_claims", []),
                    "unverified_claims": result.get("unverified_claims", []),
                },
            )

        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return JudgeResult(
                score=0.0,
                reason=f"Evaluation failed: {e}",
                passed=False,
            )

    async def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Evaluate if the answer is relevant to the question.

        Relevancy measures whether the answer directly and appropriately
        addresses what was asked in the question.

        Args:
            question: The original question asked.
            answer: The generated answer to evaluate.

        Returns:
            JudgeResult with relevancy score and explanation.
        """
        prompt = f"""You are an impartial judge evaluating the relevancy of an AI assistant's answer.

**Task:** Determine if the answer appropriately addresses the question.

**Question:** {question}

**Assistant's Answer:** {answer}

**Evaluation Criteria:**
- Score 1.0: Answer directly and completely addresses the question
- Score 0.8-0.9: Answer addresses main points, minor aspects may be missing
- Score 0.5-0.7: Answer partially addresses the question
- Score 0.2-0.4: Answer is tangentially related but doesn't address the question
- Score 0.0-0.1: Answer is completely off-topic

**Consider:**
1. Does the answer address what was specifically asked?
2. Is the answer appropriately detailed (not too brief, not too verbose)?
3. Does the answer stay focused on the question?
4. Are all parts of a multi-part question addressed?

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation of the score>",
    "addressed_aspects": ["<aspects of the question that were addressed>"],
    "missing_aspects": ["<aspects of the question that were not addressed>"]
}}"""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            result = json.loads(response.choices[0].message.content or "{}")
            score = float(result.get("score", 0.0))

            return JudgeResult(
                score=score,
                reason=result.get("reason", "No reason provided"),
                passed=score >= self._relevancy_threshold,
                details={
                    "addressed_aspects": result.get("addressed_aspects", []),
                    "missing_aspects": result.get("missing_aspects", []),
                },
            )

        except Exception as e:
            logger.error(f"Answer relevancy evaluation failed: {e}")
            return JudgeResult(
                score=0.0,
                reason=f"Evaluation failed: {e}",
                passed=False,
            )

    async def check_topic_coverage(
        self,
        answer: str,
        expected_topic: str,
    ) -> bool:
        """Check if the answer covers an expected topic.

        Uses semantic understanding rather than exact string matching,
        which is important for paraphrased or differently-worded content.

        Args:
            answer: The generated answer to check.
            expected_topic: The topic that should be covered.

        Returns:
            True if the topic is semantically covered in the answer.
        """
        prompt = f"""Does the following answer discuss or cover the topic "{expected_topic}"?

Answer: {answer}

Consider semantic meaning, not just exact word matches. The topic may be
discussed using different words or phrases.

Respond with only "yes" or "no"."""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            content = response.choices[0].message.content or ""
            return content.strip().lower() == "yes"

        except Exception as e:
            logger.error(f"Topic coverage check failed: {e}")
            return False

    async def evaluate_uncertainty_handling(
        self,
        answer: str,
        has_sufficient_context: bool,
    ) -> JudgeResult:
        """Evaluate if the answer appropriately handles uncertainty.

        When context is insufficient, the answer should acknowledge
        limitations rather than making up information.

        Args:
            answer: The generated answer to evaluate.
            has_sufficient_context: Whether the context was sufficient.

        Returns:
            JudgeResult indicating appropriate uncertainty handling.
        """
        prompt = f"""Evaluate if this answer appropriately handles uncertainty.

**Answer:** {answer}

**Context sufficiency:** {"Sufficient context was provided" if has_sufficient_context else "Context was insufficient or missing"}

**Evaluation:**
- If context was sufficient: Answer should be confident and direct
- If context was insufficient: Answer should acknowledge limitations

**Look for uncertainty indicators:**
- "I don't have information about..."
- "Based on the available context..."
- "I cannot determine..."
- 「不明です」「わかりません」「情報がありません」

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation>",
    "has_appropriate_uncertainty": <true/false>,
    "uncertainty_phrases_found": ["<list of uncertainty phrases in the answer>"]
}}"""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            result = json.loads(response.choices[0].message.content or "{}")
            score = float(result.get("score", 0.0))

            return JudgeResult(
                score=score,
                reason=result.get("reason", "No reason provided"),
                passed=score >= 0.7,
                details={
                    "has_appropriate_uncertainty": result.get(
                        "has_appropriate_uncertainty", False
                    ),
                    "uncertainty_phrases_found": result.get(
                        "uncertainty_phrases_found", []
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Uncertainty evaluation failed: {e}")
            return JudgeResult(
                score=0.0,
                reason=f"Evaluation failed: {e}",
                passed=False,
            )
