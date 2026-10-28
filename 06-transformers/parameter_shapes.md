# Transformer Tensor Shapes: A Quick Reference

Shapes make the transformer equations much easier to reason about. This page uses a simplified single-sequence view and omits batch dimensions unless stated.

Let:
- B = batch size;
- T = sequence length;
- D = model dimension;
- H = number of attention heads;
- Dh = per-head dimension, commonly D divided by H.

## Token representation

A batch of token IDs often has shape (B, T). After embedding lookup, token representations commonly have shape (B, T, D).

Position information is combined with these representations according to the architecture.

## Query, key, and value

Linear projections transform the final dimension:
- Q: (B, T, Dh) for one head;
- K: (B, T, Dh);
- V: (B, T, Dh) in a typical equal-width head.

For one head, Q multiplied by the transpose of K along the last two axes gives attention scores of shape (B, T, T). Every query position receives a score for each key position.

Softmax is applied along the key-position axis, producing weights with shape (B, T, T). Multiplying those weights by V gives output shape (B, T, Dh).

## Multiple heads

Each head uses its own projections. The head outputs are commonly concatenated along the feature dimension, returning a shape such as (B, T, H * Dh), and then projected back to (B, T, D).

When H * Dh equals D, concatenation directly restores the model width before the output projection.

## Feed-forward layer

A position-wise feed-forward network usually expands the feature width, applies a nonlinearity, and projects back:
- input: (B, T, D);
- hidden: (B, T, Dff);
- output: (B, T, D).

The time and batch dimensions remain unchanged by these per-position projections.

## Causal mask

A causal mask has logical shape (T, T) or a broadcastable equivalent. It prevents query position i from using key positions later than i. In batch implementations, additional dimensions may be added for broadcasting.

## Shape debugging checklist

1. Write down the shape of every tensor before multiplying.
2. Verify the inner dimensions match for matrix multiplication.
3. Check which axis softmax normalizes.
4. Confirm whether batch and head axes need broadcasting.
5. Print shapes on a tiny example before scaling up.
6. Treat padding and causal masks as part of model correctness, not optional decoration.
