"""Golden Q&A dataset for LLM quality evaluation.

This dataset contains question-context-answer triples for evaluating
the quality of Generation Service outputs using LLM-as-a-Judge patterns.
"""

from dataclasses import dataclass


@dataclass
class GoldenQACase:
    """A single Q&A test case for evaluation."""

    id: str
    language: str
    question: str
    context: list[str]
    expected_topics: list[str]
    must_not_contain: list[str]
    description: str = ""


GOLDEN_QA_DATASET: list[GoldenQACase] = [
    # English cases
    GoldenQACase(
        id="qa-en-001",
        language="en",
        description="Basic question about Python",
        question="What is Python?",
        context=[
            "Python is a high-level, general-purpose programming language.",
            "Python was created by Guido van Rossum and first released in 1991.",
            "Python emphasizes code readability with its notable use of significant whitespace.",
            "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        ],
        expected_topics=[
            "programming language",
            "high-level",
            "Guido van Rossum",
            "readability",
        ],
        must_not_contain=[
            "C++ is better",
            "Java is superior",
        ],
    ),
    GoldenQACase(
        id="qa-en-002",
        language="en",
        description="Question about RAG architecture",
        question="What is RAG and how does it work?",
        context=[
            "RAG (Retrieval-Augmented Generation) is an AI framework that combines information retrieval with text generation.",
            "In RAG systems, relevant documents are first retrieved from a knowledge base using semantic search.",
            "The retrieved documents are then used as context for a large language model to generate responses.",
            "RAG helps reduce hallucinations by grounding LLM responses in factual, retrieved information.",
        ],
        expected_topics=[
            "retrieval",
            "generation",
            "documents",
            "context",
            "LLM",
        ],
        must_not_contain=[],
    ),
    GoldenQACase(
        id="qa-en-003",
        language="en",
        description="Technical question about embeddings",
        question="What are embeddings in machine learning?",
        context=[
            "Embeddings are dense vector representations of data in a continuous vector space.",
            "In NLP, word embeddings map words to vectors where semantically similar words are close together.",
            "Modern embedding models like text-embedding-3-small produce 1536-dimensional vectors.",
            "Embeddings enable semantic similarity search by comparing vector distances.",
        ],
        expected_topics=[
            "vector",
            "representation",
            "similarity",
            "semantic",
        ],
        must_not_contain=[],
    ),
    # Japanese cases
    GoldenQACase(
        id="qa-ja-001",
        language="ja",
        description="Pythonに関する基本的な質問（日本語）",
        question="Pythonとは何ですか？",
        context=[
            "Pythonは高水準の汎用プログラミング言語です。",
            "Pythonは1991年にGuido van Rossumによって作成されました。",
            "Pythonは可読性の高いコードを重視し、インデントを使用してコードブロックを定義します。",
            "Pythonは手続き型、オブジェクト指向、関数型など複数のプログラミングパラダイムをサポートしています。",
        ],
        expected_topics=[
            "プログラミング言語",
            "高水準",
            "Guido van Rossum",
            "可読性",
        ],
        must_not_contain=[
            "JavaがPythonより優れている",
        ],
    ),
    GoldenQACase(
        id="qa-ja-002",
        language="ja",
        description="RAGに関する質問（日本語）",
        question="RAGとは何ですか？どのように動作しますか？",
        context=[
            "RAG（Retrieval-Augmented Generation）は、情報検索とテキスト生成を組み合わせたAIフレームワークです。",
            "RAGシステムでは、まず意味検索を使用してナレッジベースから関連文書を取得します。",
            "取得した文書は、大規模言語モデルが応答を生成するためのコンテキストとして使用されます。",
            "RAGは、LLMの応答を取得した事実に基づかせることで、ハルシネーションを減少させます。",
        ],
        expected_topics=[
            "検索",
            "生成",
            "文書",
            "コンテキスト",
            "LLM",
        ],
        must_not_contain=[],
    ),
    # Edge cases
    GoldenQACase(
        id="qa-edge-001",
        language="en",
        description="Question with limited context",
        question="What is the capital of France?",
        context=[
            "France is a country in Western Europe.",
        ],
        expected_topics=[],  # Context doesn't contain the answer
        must_not_contain=[
            "Paris is the capital",  # Should not hallucinate if context lacks this
        ],
    ),
    GoldenQACase(
        id="qa-edge-002",
        language="en",
        description="Complex multi-part question",
        question="What are the benefits and drawbacks of using microservices architecture?",
        context=[
            "Microservices architecture structures an application as a collection of loosely coupled services.",
            "Benefits include independent deployment, technology flexibility, and easier scaling of individual components.",
            "Drawbacks include increased operational complexity, network latency, and challenges in data consistency.",
            "Each microservice can be developed, deployed, and scaled independently.",
        ],
        expected_topics=[
            "benefits",
            "drawbacks",
            "independent",
            "complexity",
        ],
        must_not_contain=[],
    ),
]


def get_dataset_by_language(language: str) -> list[GoldenQACase]:
    """Filter dataset by language."""
    return [case for case in GOLDEN_QA_DATASET if case.language == language]


def get_dataset_by_id(case_id: str) -> GoldenQACase | None:
    """Get a specific test case by ID."""
    for case in GOLDEN_QA_DATASET:
        if case.id == case_id:
            return case
    return None
