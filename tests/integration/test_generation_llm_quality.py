"""LLM quality tests for Generation Service using LLM-as-a-Judge.

These tests evaluate the quality of generated answers using:
1. Custom LLM Judge (lightweight, fast)
2. Ragas metrics (comprehensive RAG evaluation)

To run these tests:
    pytest tests/integration/test_generation_llm_quality.py -m llm_quality

Requirements:
    - OPENAI_API_KEY environment variable
    - For Ragas: uv sync --extra eval
"""

import os
from uuid import uuid4

import pytest

from config.settings import OpenAISettings
from src.application.services.generation_service import GenerationService
from src.domain.entities import Chunk, Query, SearchResult
from src.infrastructure.llm.openai_llm import OpenAILLMService
from tests.integration.evaluators.llm_judge import LLMJudge
from tests.integration.evaluators.ragas_wrapper import is_ragas_available
from tests.integration.fixtures.golden_qa_dataset import (
    GOLDEN_QA_DATASET,
    GoldenQACase,
    get_dataset_by_language,
)

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm_quality,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_settings() -> OpenAISettings:
    """Create OpenAI settings for testing."""
    return OpenAISettings(
        api_key=os.environ.get("OPENAI_API_KEY", ""),  # type: ignore[arg-type]
        model="gpt-4o",
        fallback_model="gpt-4o-mini",
        timeout=60.0,
    )


@pytest.fixture
async def llm_service(openai_settings: OpenAISettings) -> OpenAILLMService:
    """Create LLM service for testing."""
    service = OpenAILLMService(openai_settings)
    yield service
    await service.close()


@pytest.fixture
def generation_service(llm_service: OpenAILLMService) -> GenerationService:
    """Create Generation service for testing."""
    return GenerationService(
        llm_service=llm_service,
        max_context_length=4000,
    )


@pytest.fixture
async def llm_judge(openai_settings: OpenAISettings) -> LLMJudge:
    """Create LLM Judge for evaluation."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_settings.api_key.get_secret_value())
    return LLMJudge(
        client=client,
        model="gpt-4o-mini",  # Use cheaper model for judging
    )


@pytest.fixture
def golden_dataset() -> list[GoldenQACase]:
    """Return the golden Q&A dataset."""
    return GOLDEN_QA_DATASET


def _create_search_results(contexts: list[str]) -> list[SearchResult]:
    """Create mock search results from context strings."""
    results = []
    doc_id = uuid4()

    for i, context in enumerate(contexts):
        chunk = Chunk.create(
            document_id=doc_id,
            content=context,
            chunk_index=i,
            start_char=0,
            end_char=len(context),
            metadata={"filename": f"test_doc_{i}.txt"},
        )
        results.append(
            SearchResult(
                chunk=chunk,
                score=0.9 - (i * 0.05),  # Decreasing scores
                rerank_score=0.95 - (i * 0.05),
                rank=i + 1,
            )
        )

    return results


# =============================================================================
# LLM Judge Tests (Fast, Custom Evaluation)
# =============================================================================


@pytest.mark.asyncio
class TestGenerationQualityWithLLMJudge:
    """Test generation quality using custom LLM Judge."""

    async def test_faithfulness_english(
        self,
        generation_service: GenerationService,
        llm_judge: LLMJudge,
    ) -> None:
        """Test faithfulness score for English Q&A pairs."""
        test_cases = get_dataset_by_language("en")[:3]  # Test first 3 English cases

        for test_case in test_cases:
            query = Query.create(text=test_case.question)
            search_results = _create_search_results(test_case.context)

            result, _ = await generation_service.generate(query, search_results)

            # Evaluate faithfulness
            judge_result = await llm_judge.evaluate_faithfulness(
                question=test_case.question,
                answer=result.answer,
                context=test_case.context,
            )

            assert judge_result.passed, (
                f"Faithfulness failed for {test_case.id}: "
                f"score={judge_result.score:.2f}, reason={judge_result.reason}"
            )

    async def test_faithfulness_japanese(
        self,
        generation_service: GenerationService,
        llm_judge: LLMJudge,
    ) -> None:
        """Test faithfulness score for Japanese Q&A pairs."""
        test_cases = get_dataset_by_language("ja")

        for test_case in test_cases:
            query = Query.create(text=test_case.question)
            search_results = _create_search_results(test_case.context)

            result, _ = await generation_service.generate(query, search_results)

            judge_result = await llm_judge.evaluate_faithfulness(
                question=test_case.question,
                answer=result.answer,
                context=test_case.context,
            )

            assert judge_result.passed, (
                f"Japanese faithfulness failed for {test_case.id}: "
                f"score={judge_result.score:.2f}, reason={judge_result.reason}"
            )

    async def test_answer_relevancy(
        self,
        generation_service: GenerationService,
        llm_judge: LLMJudge,
        golden_dataset: list[GoldenQACase],
    ) -> None:
        """Test answer relevancy across all test cases."""
        for test_case in golden_dataset[:5]:  # Test first 5 cases
            query = Query.create(text=test_case.question)
            search_results = _create_search_results(test_case.context)

            result, _ = await generation_service.generate(query, search_results)

            judge_result = await llm_judge.evaluate_answer_relevancy(
                question=test_case.question,
                answer=result.answer,
            )

            assert judge_result.passed, (
                f"Answer relevancy failed for {test_case.id}: "
                f"score={judge_result.score:.2f}, reason={judge_result.reason}"
            )

    async def test_topic_coverage(
        self,
        generation_service: GenerationService,
        llm_judge: LLMJudge,
    ) -> None:
        """Test that expected topics are covered in answers."""
        test_case = GOLDEN_QA_DATASET[0]  # Python question

        query = Query.create(text=test_case.question)
        search_results = _create_search_results(test_case.context)

        result, _ = await generation_service.generate(query, search_results)

        # Check at least some expected topics are covered
        covered_count = 0
        for topic in test_case.expected_topics:
            if await llm_judge.check_topic_coverage(result.answer, topic):
                covered_count += 1

        # At least 50% of expected topics should be covered
        coverage_ratio = covered_count / len(test_case.expected_topics)
        assert coverage_ratio >= 0.5, (
            f"Topic coverage too low: {coverage_ratio:.0%} "
            f"({covered_count}/{len(test_case.expected_topics)})"
        )

    async def test_no_hallucination(
        self,
        generation_service: GenerationService,
        golden_dataset: list[GoldenQACase],
    ) -> None:
        """Test that answers don't contain forbidden hallucinated content."""
        for test_case in golden_dataset:
            if not test_case.must_not_contain:
                continue

            query = Query.create(text=test_case.question)
            search_results = _create_search_results(test_case.context)

            result, _ = await generation_service.generate(query, search_results)

            for forbidden in test_case.must_not_contain:
                assert forbidden.lower() not in result.answer.lower(), (
                    f"Hallucination detected in {test_case.id}: "
                    f"'{forbidden}' found in answer"
                )

    async def test_empty_context_handling(
        self,
        generation_service: GenerationService,
        llm_judge: LLMJudge,
    ) -> None:
        """Test appropriate handling when context is empty or insufficient."""
        query = Query.create(text="What is the population of Mars?")

        result, _ = await generation_service.generate_with_no_context(query)

        # Should express uncertainty appropriately
        judge_result = await llm_judge.evaluate_uncertainty_handling(
            answer=result.answer,
            has_sufficient_context=False,
        )

        assert judge_result.passed, (
            f"Uncertainty handling failed: "
            f"score={judge_result.score:.2f}, reason={judge_result.reason}"
        )


# =============================================================================
# Ragas Tests (Comprehensive RAG Metrics)
# =============================================================================


@pytest.mark.skipif(not is_ragas_available(), reason="Ragas not installed")
@pytest.mark.asyncio
class TestGenerationQualityWithRagas:
    """Test generation quality using Ragas metrics.

    These tests require the 'eval' extra dependency:
        uv sync --extra eval
    """

    async def test_ragas_faithfulness(
        self,
        generation_service: GenerationService,
    ) -> None:
        """Test faithfulness using Ragas metric."""
        from tests.integration.evaluators.ragas_wrapper import RagasEvaluator

        evaluator = RagasEvaluator()
        test_case = GOLDEN_QA_DATASET[0]

        query = Query.create(text=test_case.question)
        search_results = _create_search_results(test_case.context)

        result, _ = await generation_service.generate(query, search_results)

        scores = await evaluator.evaluate(
            question=test_case.question,
            answer=result.answer,
            contexts=test_case.context,
        )

        assert scores.faithfulness >= 0.8, (
            f"Ragas faithfulness too low: {scores.faithfulness:.2f}"
        )

    async def test_ragas_answer_relevancy(
        self,
        generation_service: GenerationService,
    ) -> None:
        """Test answer relevancy using Ragas metric."""
        from tests.integration.evaluators.ragas_wrapper import RagasEvaluator

        evaluator = RagasEvaluator()
        test_case = GOLDEN_QA_DATASET[0]

        query = Query.create(text=test_case.question)
        search_results = _create_search_results(test_case.context)

        result, _ = await generation_service.generate(query, search_results)

        scores = await evaluator.evaluate(
            question=test_case.question,
            answer=result.answer,
            contexts=test_case.context,
        )

        assert scores.answer_relevancy >= 0.7, (
            f"Ragas answer relevancy too low: {scores.answer_relevancy:.2f}"
        )

    async def test_ragas_batch_evaluation(
        self,
        generation_service: GenerationService,
    ) -> None:
        """Test batch evaluation using Ragas."""
        from tests.integration.evaluators.ragas_wrapper import RagasEvaluator

        evaluator = RagasEvaluator()
        test_cases = GOLDEN_QA_DATASET[:3]  # First 3 cases

        questions = []
        answers = []
        contexts_list = []

        for test_case in test_cases:
            query = Query.create(text=test_case.question)
            search_results = _create_search_results(test_case.context)
            result, _ = await generation_service.generate(query, search_results)

            questions.append(test_case.question)
            answers.append(result.answer)
            contexts_list.append(test_case.context)

        scores_list = await evaluator.evaluate_batch(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
        )

        # Check all pass thresholds
        for i, scores in enumerate(scores_list):
            passed, failures = evaluator.check_thresholds(scores)
            assert passed, f"Test case {test_cases[i].id} failed: {', '.join(failures)}"


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.asyncio
class TestStreamingGenerationQuality:
    """Test quality of streaming generation."""

    async def test_streaming_produces_coherent_output(
        self,
        generation_service: GenerationService,
        llm_judge: LLMJudge,
    ) -> None:
        """Test that streaming output is coherent when assembled."""
        test_case = GOLDEN_QA_DATASET[0]

        query = Query.create(text=test_case.question)
        search_results = _create_search_results(test_case.context)

        # Collect streaming chunks
        chunks = []
        async for chunk in generation_service.generate_stream(query, search_results):
            chunks.append(chunk)

        # Assemble full answer
        full_answer = "".join(chunks)

        # Verify it's coherent and relevant
        judge_result = await llm_judge.evaluate_answer_relevancy(
            question=test_case.question,
            answer=full_answer,
        )

        assert judge_result.passed, (
            f"Streaming output not coherent: {judge_result.reason}"
        )


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.asyncio
class TestGenerationPerformance:
    """Test generation performance metrics."""

    async def test_generation_latency(
        self,
        generation_service: GenerationService,
    ) -> None:
        """Test that generation completes within acceptable latency."""
        test_case = GOLDEN_QA_DATASET[0]

        query = Query.create(text=test_case.question)
        search_results = _create_search_results(test_case.context)

        _, metrics = await generation_service.generate(query, search_results)

        # Total latency should be under 10 seconds
        assert metrics.total_latency_ms < 10000, (
            f"Generation too slow: {metrics.total_latency_ms:.0f}ms"
        )

    async def test_token_usage_tracking(
        self,
        generation_service: GenerationService,
    ) -> None:
        """Test that token usage is properly tracked."""
        test_case = GOLDEN_QA_DATASET[0]

        query = Query.create(text=test_case.question)
        search_results = _create_search_results(test_case.context)

        result, metrics = await generation_service.generate(query, search_results)

        # Token usage should be non-zero
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens > 0

        # Context tokens estimate should be reasonable
        assert metrics.context_tokens_estimate > 0
