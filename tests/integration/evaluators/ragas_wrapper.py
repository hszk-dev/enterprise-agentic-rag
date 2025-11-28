"""Ragas evaluation wrapper for RAG quality assessment.

This module provides a wrapper around Ragas metrics for evaluating
RAG system quality. It simplifies the Ragas API for our specific use case.

Requires: pip install ragas datasets
(Available via: uv sync --extra eval)

Note: This wrapper is compatible with Ragas 0.3.x API.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Ragas imports - these are optional dependencies
try:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "Ragas not installed. Run 'uv sync --extra eval' to enable Ragas metrics."
    )


@dataclass
class RagasScores:
    """Collection of Ragas evaluation scores.

    Attributes:
        faithfulness: How faithful the answer is to the context (0-1).
        answer_relevancy: How relevant the answer is to the question (0-1).
    """

    faithfulness: float
    answer_relevancy: float

    @property
    def average_score(self) -> float:
        """Calculate average of available scores."""
        return (self.faithfulness + self.answer_relevancy) / 2


class RagasEvaluator:
    """Wrapper for Ragas RAG evaluation metrics.

    Provides a simplified interface to Ragas 0.3.x evaluation metrics,
    focusing on the metrics most relevant for RAG quality assessment.

    Example:
        >>> evaluator = RagasEvaluator()
        >>> scores = await evaluator.evaluate(
        ...     question="What is Python?",
        ...     answer="Python is a programming language.",
        ...     contexts=["Python is a high-level programming language."],
        ... )
        >>> print(f"Faithfulness: {scores.faithfulness}")
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.8,
        relevancy_threshold: float = 0.7,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the Ragas evaluator.

        Args:
            faithfulness_threshold: Minimum faithfulness score to pass.
            relevancy_threshold: Minimum answer relevancy score to pass.
            model: OpenAI model to use for evaluation.

        Raises:
            ImportError: If Ragas is not installed.
        """
        if not RAGAS_AVAILABLE:
            msg = "Ragas not installed. Install with: uv sync --extra eval"
            raise ImportError(msg)

        self._faithfulness_threshold = faithfulness_threshold
        self._relevancy_threshold = relevancy_threshold
        self._model = model
        self._llm = self._create_llm()

    def _create_llm(self) -> "LangchainLLMWrapper":
        """Create LLM wrapper for Ragas metrics."""
        from langchain_openai import ChatOpenAI

        # Ragas 0.3.x uses LangchainLLMWrapper
        # Note: llm_factory is recommended but requires more setup
        chat_llm = ChatOpenAI(
            model=self._model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        return LangchainLLMWrapper(chat_llm)

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RagasScores:
        """Evaluate a single Q&A pair using Ragas metrics.

        Args:
            question: The question that was asked.
            answer: The generated answer.
            contexts: List of context passages used for generation.

        Returns:
            RagasScores with evaluation results.
        """
        # Create sample in Ragas 0.3.x format
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])

        # Create metrics with LLM
        metrics = [
            Faithfulness(llm=self._llm),
            ResponseRelevancy(llm=self._llm),
        ]

        # Run evaluation
        result = evaluate(dataset, metrics=metrics)

        # Extract scores from result
        # result.scores is a list of dicts, one per sample
        sample_scores = result.scores[0] if result.scores else {}

        scores = RagasScores(
            faithfulness=float(sample_scores.get("faithfulness", 0.0)),
            answer_relevancy=float(sample_scores.get("answer_relevancy", 0.0)),
        )

        return scores

    async def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
    ) -> list[RagasScores]:
        """Evaluate multiple Q&A pairs in batch.

        More efficient than calling evaluate() multiple times
        due to batched API calls.

        Args:
            questions: List of questions.
            answers: List of generated answers.
            contexts_list: List of context lists (one per question).

        Returns:
            List of RagasScores for each Q&A pair.
        """
        if len(questions) != len(answers) or len(questions) != len(contexts_list):
            msg = "All input lists must have the same length"
            raise ValueError(msg)

        # Create samples in Ragas 0.3.x format
        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
            )
            for q, a, c in zip(questions, answers, contexts_list, strict=True)
        ]
        dataset = EvaluationDataset(samples=samples)

        # Create metrics with LLM
        metrics = [
            Faithfulness(llm=self._llm),
            ResponseRelevancy(llm=self._llm),
        ]

        # Run batch evaluation
        result = evaluate(dataset, metrics=metrics)

        # Convert to list of RagasScores
        scores_list = []
        for sample_scores in result.scores:
            scores_list.append(
                RagasScores(
                    faithfulness=float(sample_scores.get("faithfulness", 0.0)),
                    answer_relevancy=float(sample_scores.get("answer_relevancy", 0.0)),
                )
            )

        return scores_list

    def check_thresholds(self, scores: RagasScores) -> tuple[bool, list[str]]:
        """Check if scores meet the configured thresholds.

        Args:
            scores: RagasScores to check.

        Returns:
            Tuple of (all_passed, list_of_failures).
        """
        failures = []

        if scores.faithfulness < self._faithfulness_threshold:
            failures.append(
                f"Faithfulness {scores.faithfulness:.2f} < {self._faithfulness_threshold}"
            )

        if scores.answer_relevancy < self._relevancy_threshold:
            failures.append(
                f"Answer relevancy {scores.answer_relevancy:.2f} < {self._relevancy_threshold}"
            )

        return len(failures) == 0, failures


def is_ragas_available() -> bool:
    """Check if Ragas is available for use."""
    return RAGAS_AVAILABLE
