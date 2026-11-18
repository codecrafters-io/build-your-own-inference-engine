In this stage, you'll implement the `logits` command to score every token as
the next one.

### Why logits?

The blocks change each row, but they don't say which token comes next. Logits
do. A logit is the model's raw score for one token as the next token, and a
higher logit means a more likely next token. These scores are not probabilities.
Softmax can turn them into probabilities, as it did for attention scores, but
this command returns the raw scores.

GPT-2 has 50,257 tokens, so it writes 50,257 logits, one per token. These are
the five highest after "The capital of France is":

| Token | Logit |
| --- | --- |
| " the" | -100.25 |
| " now" | -100.82 |
| " a" | -100.86 |
| " France" | -101.21 |
| " Paris" | -101.21 |

These leading logits are near -100. Negative scores are valid: only their
differences affect the softmax probabilities, so " the" leads " now" by 0.57.

### The `logits` command

`logits` reads three files:

- `model.safetensors`, with every tensor `embed` and `block` read, plus a final
  layer norm:
    - `ln_f.weight`, with shape `[n_embd]`
    - `ln_f.bias`, with shape `[n_embd]`
- `config.json`, the same format as for `attn`
- `ids.json`, the same format as for `embed`

The IDs pass through embedding and every transformer block, in order from
layer 0 through `n_layer - 1`. This computation is called a forward pass.
The last position's row contains the context for predicting the next token.
Apply `ln_f` to that row using the same layer-normalization formula and
0.00001 constant as in a block.

Each token's logit is the dot product of that final row with the token's row
in `wte.weight`. Return the scores in token-ID order.

The logits reuse the `wte.weight` table that `embed` reads. There is no
separate output table.

In this example the ids are `[0]`. `ln_f.bias` is `[1, -1]` and `ln_f.weight`
is `[0, 0]`, so the normalized row is `[1, -1]`. The token rows are `[1, 1]` and
`[1, -1]`. The dot products are `1 * 1 + -1 * 1 = 0` and
`1 * 1 + -1 * -1 = 2`.

Print one JSON array, the scores for the last position:

```json
[0.0, 2.0]
```

### Tests

The tester will write the three files and execute your program like this:

```bash
$ ./your_program.sh logits model.safetensors config.json ids.json
```

Your program must print:

```json
[0.0, 2.0]
```

The tester allows a difference of 0.001 on each number, or 0.001 times the
absolute value of the expected number, whichever is larger. It will also run
two models whose weights are random numbers between -1 and 1, each with six
tokens and a three-id sequence, where only the last position is scored. One
has a single layer, and the other has two.
