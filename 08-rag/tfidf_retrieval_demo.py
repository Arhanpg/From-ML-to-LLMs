"""A tiny lexical retrieval example using TF-IDF and cosine similarity.

This demonstrates only the retrieval stage of a RAG pipeline. It does not call
an LLM or claim to generate an answer.

Run from the repository root:
    python 08-rag/tfidf_retrieval_demo.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = [
    {
        "id": "doc-1",
        "title": "Train and test data",
        "text": "Split data before fitting preprocessing. Fit transformations on training data only.",
    },
    {
        "id": "doc-2",
        "title": "Regression metrics",
        "text": "Mean absolute error is measured in target units. Root mean squared error penalizes large residuals.",
    },
    {
        "id": "doc-3",
        "title": "Transformer attention",
        "text": "Scaled dot product attention combines query, key, and value representations.",
    },
    {
        "id": "doc-4",
        "title": "RAG evaluation",
        "text": "Evaluate retrieval separately from answer generation and verify that citations support claims.",
    },
    {
        "id": "doc-5",
        "title": "Data leakage",
        "text": "Data leakage occurs when information unavailable at prediction time influences model development.",
    },
]


def retrieve(query: str, top_k: int = 3) -> list[tuple[dict[str, str], float]]:
    """Rank documents by TF-IDF cosine similarity to the query."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least one")

    corpus = [document["text"] + " " + document["title"] for document in DOCUMENTS]
    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors)[0]

    ranked_indices = sorted(
        range(len(DOCUMENTS)), key=lambda index: scores[index], reverse=True
    )
    return [
        (DOCUMENTS[index], float(scores[index]))
        for index in ranked_indices[: min(top_k, len(DOCUMENTS))]
    ]


def main() -> None:
    query = "How can I avoid leaking test data during preprocessing?"
    print("Query:", query)
    print("\nRetrieved passages:")
    for rank, (document, score) in enumerate(retrieve(query, top_k=3), start=1):
        print(f"{rank}. {document['title']} ({document['id']}) | score={score:.3f}")
        print(f"   {document['text']}")

    print(
        "\nTF-IDF is lexical retrieval. A real RAG system also needs authorization, "
        "context assembly, grounded generation, citations, and evaluation."
    )


if __name__ == "__main__":
    main()
