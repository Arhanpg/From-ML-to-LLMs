# Self-Attention: Exercises and Shape Reasoning

These exercises accompany README.md and self_attention_numpy.py.

## Shape walkthrough

Suppose a sequence has 5 positions and each input representation has 8 dimensions.

- Input matrix X has shape (5, 8).
- If query/key projections have depth 4, Q and K each have shape (5, 4).
- The score matrix Q times the transpose of K has shape (5, 5).
- Each row contains one query position's scores against all key positions.
- If V has shape (5, 6), multiplying attention weights by V produces shape (5, 6).

The output feature dimension follows V, not Q.

## Exercise 1: row normalization

Run self_attention_numpy.py and inspect the attention weights. Explain why each row sums approximately to one. Why might floating-point arithmetic mean the printed sum is not exactly 1.0?

## Exercise 2: causal masking

A causal attention mask allows position i to use positions 0 through i, but not positions after i. Write a 4-by-4 lower-triangular boolean mask. Explain which positions are visible to row 2 using zero-based indexing.

## Exercise 3: scaling

The attention scores are divided by the square root of the key depth. What might happen to softmax values if dot products grow in magnitude with dimension and the scores are not scaled?

## Exercise 4: multiple heads

Suppose model dimension is 12 and attention uses 3 heads with per-head depth 4. Explain how head outputs can be combined to return to model dimension 12. Why may multiple heads learn different relationships?

## Exercise 5: what attention does not prove

A particular token receives a high attention weight from another token. Why is this not, by itself, a complete causal explanation for the final prediction? Consider other layers, residual paths, the value vectors, and output projection.

## Exercise 6: padding

A batch contains sequences of different lengths, padded to the same length. Explain why padded positions should not contribute as valid keys and describe how a padding mask can help.
