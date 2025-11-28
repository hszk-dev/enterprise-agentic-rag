# Generation Service 統合テスト設計

**作成日:** 2025-11-29
**対象PR:** #13 (Step 5: Generation - LLM Integration)

## 1. 背景と課題

### 1.1 LLM応答の非決定性

LLMの回答は非決定的であるため、従来のテストフレームワーク（完全一致比較など）では評価が困難：

- 同じ入力でも異なる出力が生成される
- 「正解」が複数存在しうる
- 文体や表現の違いが許容される

### 1.2 解決アプローチ: LLM as a Judge

LLM自身を評価者として使用する「LLM as a Judge」パターンを採用：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│ Generation  │────▶│  LLM Judge  │
│  (input)    │     │  Service    │     │ (evaluator) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Answer    │     │   Score &   │
                    │  (output)   │     │   Reason    │
                    └─────────────┘     └─────────────┘
```

## 2. 評価フレームワーク選定

### 2.1 選定: Ragas

| フレームワーク | メリット | デメリット | 採用 |
|---------------|----------|------------|------|
| **Ragas** | RAG専用、RAG Triadメトリクス | 設定が複雑 | ✅ |
| DeepEval | pytest統合、汎用 | RAG特化ではない | - |
| 独自実装 | 柔軟性 | メンテナンスコスト | - |

**採用理由:**
- 既に `pyproject.toml` の `eval` 依存関係に含まれている
- RAG専用の評価メトリクス（RAG Triad）を提供
- 学術研究での実績（WikiEval等）

### 2.2 RAG Triad メトリクス

```
                    ┌─────────────────────────────────┐
                    │         Answer Relevancy        │
                    │   (回答が質問に対して適切か)      │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────┐ │
│  │   Query     │─────────▶│   Context   │─────────▶│ Answer  │ │
│  └─────────────┘          └─────────────┘          └─────────┘ │
│         │                        │                       │      │
│         │                        │                       │      │
│         ▼                        ▼                       │      │
│  ┌─────────────────┐   ┌─────────────────┐              │      │
│  │Context Relevancy│   │  Faithfulness   │◀─────────────┘      │
│  │(検索結果が質問  │   │(回答がコンテキ  │                      │
│  │ に関連してるか) │   │ ストに基づくか) │                      │
│  └─────────────────┘   └─────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| メトリクス | 説明 | しきい値（推奨） |
|-----------|------|-----------------|
| **Faithfulness** | 回答がコンテキストに基づいているか（ハルシネーション防止） | ≥ 0.8 |
| **Answer Relevancy** | 回答が質問に対して適切に回答しているか | ≥ 0.7 |
| **Context Relevancy** | 検索結果が質問に関連しているか | ≥ 0.6 |

## 3. テスト設計

### 3.1 テストの種類

```
tests/integration/
└── test_generation_service.py      # 既存（モック使用）
└── test_generation_llm_quality.py  # 新規（LLM as a Judge）
```

### 3.2 評価データセット（ゴールデンセット）

```python
# tests/integration/fixtures/golden_qa_dataset.py

GOLDEN_QA_DATASET = [
    {
        "id": "qa-001",
        "language": "en",
        "question": "What is Python?",
        "context": [
            "Python is a high-level, general-purpose programming language.",
            "Python was created by Guido van Rossum in 1991.",
            "Python emphasizes code readability with its notable use of whitespace.",
        ],
        "expected_topics": ["programming language", "high-level", "Guido van Rossum"],
        "must_not_contain": ["C++", "Java is better"],  # ハルシネーション検出用
    },
    {
        "id": "qa-002",
        "language": "ja",
        "question": "Pythonとは何ですか？",
        "context": [
            "Pythonは高水準の汎用プログラミング言語です。",
            "Pythonは1991年にGuido van Rossumによって作成されました。",
            "Pythonは可読性の高いコードを重視しています。",
        ],
        "expected_topics": ["プログラミング言語", "高水準", "Guido van Rossum"],
        "must_not_contain": ["JavaがPythonより優れている"],
    },
    {
        "id": "qa-003",
        "language": "en",
        "question": "What is RAG?",
        "context": [
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
            "RAG systems first retrieve relevant documents, then use them as context for LLM generation.",
            "RAG helps reduce hallucinations by grounding responses in retrieved facts.",
        ],
        "expected_topics": ["retrieval", "generation", "documents", "context"],
        "must_not_contain": [],
    },
]
```

### 3.3 テストケース設計

#### Case 1: 基本的な回答品質テスト

```python
@pytest.mark.integration
@pytest.mark.llm_quality
class TestGenerationQualityWithRagas:
    """Generation service quality tests using Ragas metrics."""

    async def test_faithfulness_score(
        self,
        generation_service: GenerationService,
        golden_dataset: list[dict],
    ) -> None:
        """Test that answers are faithful to the provided context."""
        for test_case in golden_dataset:
            query = Query.create(text=test_case["question"])
            search_results = self._create_search_results(test_case["context"])

            result, _ = await generation_service.generate(query, search_results)

            # Ragas Faithfulness評価
            score = await evaluate_faithfulness(
                question=test_case["question"],
                answer=result.answer,
                contexts=test_case["context"],
            )

            assert score >= 0.8, (
                f"Faithfulness score {score:.2f} < 0.8 for {test_case['id']}"
            )

    async def test_answer_relevancy_score(
        self,
        generation_service: GenerationService,
        golden_dataset: list[dict],
    ) -> None:
        """Test that answers are relevant to the question."""
        for test_case in golden_dataset:
            query = Query.create(text=test_case["question"])
            search_results = self._create_search_results(test_case["context"])

            result, _ = await generation_service.generate(query, search_results)

            # Ragas Answer Relevancy評価
            score = await evaluate_answer_relevancy(
                question=test_case["question"],
                answer=result.answer,
            )

            assert score >= 0.7, (
                f"Answer relevancy score {score:.2f} < 0.7 for {test_case['id']}"
            )
```

#### Case 2: ハルシネーション検出テスト

```python
async def test_no_hallucination(
    self,
    generation_service: GenerationService,
    golden_dataset: list[dict],
) -> None:
    """Test that answers don't contain hallucinated content."""
    for test_case in golden_dataset:
        if not test_case.get("must_not_contain"):
            continue

        query = Query.create(text=test_case["question"])
        search_results = self._create_search_results(test_case["context"])

        result, _ = await generation_service.generate(query, search_results)

        for forbidden in test_case["must_not_contain"]:
            assert forbidden.lower() not in result.answer.lower(), (
                f"Hallucination detected: '{forbidden}' in answer for {test_case['id']}"
            )
```

#### Case 3: コンテキストなしの挙動テスト

```python
async def test_no_context_handling(
    self,
    generation_service: GenerationService,
) -> None:
    """Test that service handles empty context appropriately."""
    query = Query.create(text="What is the capital of France?")

    result, _ = await generation_service.generate_with_no_context(query)

    # コンテキストがない場合、不明であることを表明すべき
    uncertainty_indicators = [
        "don't have", "cannot", "no information", "不明", "わかりません",
        "context", "provided"
    ]

    has_uncertainty = any(
        indicator.lower() in result.answer.lower()
        for indicator in uncertainty_indicators
    )

    assert has_uncertainty, (
        "Answer should indicate uncertainty when no context is provided"
    )
```

#### Case 4: 日本語品質テスト

```python
@pytest.mark.parametrize("test_case", [
    tc for tc in GOLDEN_QA_DATASET if tc["language"] == "ja"
])
async def test_japanese_answer_quality(
    self,
    generation_service: GenerationService,
    test_case: dict,
) -> None:
    """Test Japanese answer quality."""
    query = Query.create(text=test_case["question"])
    search_results = self._create_search_results(test_case["context"])

    result, _ = await generation_service.generate(query, search_results)

    # 日本語で回答されているか
    assert contains_japanese(result.answer), "Answer should be in Japanese"

    # 期待されるトピックが含まれているか
    for topic in test_case["expected_topics"]:
        # 完全一致ではなく、意味的に含まれているかをLLMで評価
        contains = await llm_check_topic_coverage(result.answer, topic)
        assert contains, f"Expected topic '{topic}' not covered in answer"
```

### 3.4 カスタムLLM Judge実装

Ragasのデフォルト評価に加え、プロジェクト固有の評価基準を追加：

```python
# tests/integration/evaluators/llm_judge.py

from dataclasses import dataclass
from openai import AsyncOpenAI


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: float  # 0.0 - 1.0
    reason: str
    passed: bool


class LLMJudge:
    """Custom LLM-as-a-Judge evaluator."""

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> JudgeResult:
        """Evaluate if the answer is faithful to the context."""
        prompt = f"""You are an impartial judge evaluating the faithfulness of an AI assistant's answer.

Question: {question}

Context provided to the assistant:
{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(context))}

Assistant's Answer: {answer}

Evaluate whether the answer is faithful to the provided context.
- Faithful means the answer only contains information that can be verified from the context
- Unfaithful means the answer contains information not present in or contradicting the context

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation>",
    "unfaithful_claims": ["<list of claims not supported by context>"]
}}"""

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = json.loads(response.choices[0].message.content)
        return JudgeResult(
            score=result["score"],
            reason=result["reason"],
            passed=result["score"] >= 0.8,
        )

    async def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Evaluate if the answer is relevant to the question."""
        prompt = f"""You are an impartial judge evaluating the relevancy of an AI assistant's answer.

Question: {question}

Assistant's Answer: {answer}

Evaluate whether the answer directly and appropriately addresses the question.
Consider:
- Does the answer address what was asked?
- Is the answer complete (covers main aspects)?
- Is the answer appropriately detailed (not too verbose, not too brief)?

Respond in JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation>",
    "missing_aspects": ["<aspects of the question not addressed>"]
}}"""

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = json.loads(response.choices[0].message.content)
        return JudgeResult(
            score=result["score"],
            reason=result["reason"],
            passed=result["score"] >= 0.7,
        )
```

## 4. テスト実行戦略

### 4.1 テストマーカー

```python
# pytest.ini または pyproject.toml
markers = [
    "llm_quality: LLM quality tests using LLM-as-a-Judge (requires API keys)",
    "slow: Slow tests (LLM API calls)",
]
```

### 4.2 実行コマンド

```bash
# 通常の統合テスト（モック使用）
make test-integration

# LLM品質テスト（実際のAPI使用、CI/CDではスキップ可能）
pytest tests/integration/test_generation_llm_quality.py -m llm_quality

# 全テスト（開発時のみ）
pytest tests/integration/ -m "integration or llm_quality"
```

### 4.3 CI/CD統合

```yaml
# .github/workflows/test.yml
jobs:
  unit-tests:
    # 常に実行

  integration-tests:
    # 常に実行（モック使用）

  llm-quality-tests:
    # mainブランチへのマージ時、または手動トリガー時のみ
    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## 5. テストデータ管理

### 5.1 ゴールデンセットの管理

```
tests/
├── integration/
│   ├── fixtures/
│   │   ├── golden_qa_dataset.py    # ゴールデンQ&Aデータセット
│   │   ├── golden_qa_dataset.json  # JSON形式（外部ツール連携用）
│   │   └── sample.pdf              # 既存のサンプルファイル
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── llm_judge.py            # カスタムLLM Judge
│   │   └── ragas_wrapper.py        # Ragas評価ラッパー
│   └── test_generation_llm_quality.py
```

### 5.2 テスト結果の保存

```python
# テスト結果をLangfuseに送信してトラッキング
from langfuse import Langfuse

langfuse = Langfuse()

async def log_evaluation_result(
    test_id: str,
    question: str,
    answer: str,
    scores: dict[str, float],
) -> None:
    """Log evaluation results to Langfuse for tracking."""
    langfuse.score(
        name="faithfulness",
        value=scores["faithfulness"],
        trace_id=test_id,
    )
    langfuse.score(
        name="answer_relevancy",
        value=scores["answer_relevancy"],
        trace_id=test_id,
    )
```

## 6. 期待される成果物

### 6.1 新規ファイル

| ファイル | 説明 |
|---------|------|
| `tests/integration/fixtures/golden_qa_dataset.py` | 評価用Q&Aデータセット |
| `tests/integration/evaluators/llm_judge.py` | カスタムLLM Judge実装 |
| `tests/integration/evaluators/ragas_wrapper.py` | Ragas評価ラッパー |
| `tests/integration/test_generation_llm_quality.py` | LLM品質統合テスト |

### 6.2 変更ファイル

| ファイル | 変更内容 |
|---------|----------|
| `pyproject.toml` | `llm_quality` マーカー追加 |
| `tests/integration/conftest.py` | LLM Judge関連フィクスチャ追加 |

## 7. 参考資料

- [Ragas Documentation](https://docs.ragas.io/)
- [LLM as a Judge - Ragas](https://docs.ragas.io/en/stable/howtos/applications/align-llm-as-judge/)
- [AWS: Evaluate Amazon Bedrock Agents with Ragas](https://aws.amazon.com/blogs/machine-learning/evaluate-amazon-bedrock-agents-with-ragas-and-llm-as-a-judge/)
- [DeepEval - LLM Evaluation Framework](https://deepeval.com/)
- [Mistral: Evaluating RAG with LLM as a Judge](https://mistral.ai/news/llm-as-rag-judge)
