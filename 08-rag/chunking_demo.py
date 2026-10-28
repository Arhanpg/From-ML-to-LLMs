"""A simple word-window chunker for learning about RAG preprocessing.

This demonstration uses words, not a model tokenizer. Production chunking should
respect document structure and the tokenization of the target embedding/model.
"""


def chunk_words(
    text: str, chunk_size: int = 40, overlap: int = 8
) -> list[dict[str, int | str]]:
    """Create overlapping word chunks with start/end word offsets."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    chunks: list[dict[str, int | str]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(
            {
                "chunk_index": len(chunks),
                "start_word": start,
                "end_word_exclusive": end,
                "text": " ".join(words[start:end]),
            }
        )
        if end == len(words):
            break
        start = end - overlap

    return chunks


def main() -> None:
    document = (
        "Retrieval augmented generation systems first ingest documents and extract text. "
        "A chunker divides long text into smaller passages. Some overlap can preserve "
        "context across boundaries, but too much overlap increases index size and can "
        "retrieve near-duplicate passages. Chunk size should be tested against retrieval "
        "quality, source structure, context limits, and the questions users actually ask. "
        "Metadata such as document identifiers and source locations helps citations."
    )

    chunks = chunk_words(document, chunk_size=18, overlap=4)
    for chunk in chunks:
        print(
            f"Chunk {chunk['chunk_index']} "
            f"words [{chunk['start_word']}, {chunk['end_word_exclusive']}):"
        )
        print(f"  {chunk['text']}")

    print("\nThis example demonstrates chunk boundaries, not semantic or token-aware chunking.")


if __name__ == "__main__":
    main()
