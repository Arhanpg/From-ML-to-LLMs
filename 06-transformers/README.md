# 06 — Transformers and Self-Attention

Transformers are neural network architectures that use attention to combine information across elements in a sequence. In language applications, those elements commonly represent tokens.

## From text to vectors

A language-model input usually goes through several stages:

1. A tokenizer converts text into token IDs.
2. An embedding table maps token IDs to vectors.
3. Positional information helps the model distinguish token order.
4. Transformer blocks update token representations using attention and feed-forward operations.
5. A projection produces scores, often called logits, over the vocabulary.

Tokenization is not the same as splitting on spaces. A tokenizer may represent a word as one token or several subword tokens, and token IDs are model-specific.

## Query, key, and value

For an input representation matrix X, learned projections create:

    Q = X Wq
    K = X Wk
    V = X Wv

Each query describes what a position is looking for, keys describe information positions expose for matching, and values contain information to combine.

Scaled dot-product attention is:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

The dot products produce compatibility scores between positions. Scaling by the square root of the key dimension helps keep score magnitudes manageable. Softmax converts each row of scores into weights, and the weighted sum of values produces the output representation.

## Why attention masks matter

In a decoder-only autoregressive language model, a causal mask prevents a token from attending to future tokens. Without the mask, training could expose information unavailable when generating token by token.

Padding masks may also be used to keep padded positions from affecting valid tokens.

## Transformer block ingredients

A common block includes:
- attention;
- a residual connection;
- normalization;
- a position-wise feed-forward network;
- another residual connection and normalization.

Exact order and normalization choices vary by architecture. Multi-head attention uses multiple projections so the model can learn different attention patterns.

## Runnable toy calculation

Run from the repository root:

    python 06-transformers/self_attention_numpy.py

This small script constructs deterministic vectors and computes scaled dot-product attention. It is not a language model: it omits tokenization, learned training, multiple heads, masking, and transformer blocks. Its purpose is to make the matrix operations visible.

## Exercises

1. Verify that every row of attention weights sums to one within floating-point tolerance.
2. Change the query/key dimension and observe the scaling.
3. Apply a causal mask and explain which entries become unavailable.
4. Explain why attention weights are not automatically a faithful explanation of the model's full decision.
5. Compare a bidirectional encoder's attention pattern with causal attention in a decoder.
