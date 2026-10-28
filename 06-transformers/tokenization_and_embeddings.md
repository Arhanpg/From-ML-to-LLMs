# Tokenization, Embeddings, and Positional Information

Before a transformer can process text, the text must be represented numerically. Three concepts are easy to confuse: tokens, token IDs, and embeddings.

## 1. Tokens are units selected by a tokenizer

A tokenizer maps a string into a sequence of token units. Depending on its vocabulary and rules, a token may represent a whole word, part of a word, punctuation, whitespace, or another textual unit.

For example, the word "unhelpful" might be represented by one token or split into subword tokens. The exact split depends on the tokenizer; never assume tokenization is identical across models.

Token IDs are integer identifiers assigned to entries in a tokenizer's vocabulary. An ID is not itself a semantic vector.

## 2. Embeddings are vectors

An embedding table maps each token ID to a learned vector. If the vocabulary has V entries and the embedding dimension is d, the table is often conceptually shaped V by d.

The embedding vector gives the network a numerical representation that can be updated by training. Nearby vectors may encode useful relationships, but vector distance should not automatically be interpreted as a complete or human-readable explanation of meaning.

## 3. Why position matters

Attention operations alone do not inherently tell the model the original order of tokens. Transformer architectures therefore include positional information, though methods differ. Some add position vectors; others use relative-position methods or rotary position encodings.

Word order changes meaning. A representation of "dog bites person" needs to preserve sequence information differently from "person bites dog."

## 4. Contextual representations

An input embedding starts from token identity and positional information. As it passes through transformer blocks, self-attention mixes information across positions and feed-forward layers transform each position's representation.

The resulting representation of a token depends on its context. The same token can therefore have different internal representations in different sentences.

## 5. Tokenization affects application design

Token count influences:
- context-window limits;
- prompt and output budgets;
- inference cost where pricing is token-based;
- chunk sizes for retrieval systems;
- truncation behavior.

A character count or word count is not a dependable substitute for counting tokens with the target model's tokenizer.

## Exercises

1. Tokenize a sentence using a tokenizer associated with a model you are studying.
2. Compare the token count for a short sentence and its paraphrase.
3. Explain the difference between a token ID and the embedding vector looked up from that ID.
4. Explain why contextual embeddings differ for the same word in two sentences.
5. Describe one way that truncation could remove important evidence from a RAG prompt.
