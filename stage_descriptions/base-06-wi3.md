In this stage, you'll implement the `logits` command to score every token as
the next one.

### Why logits?

The blocks change each row, but they don't say which token comes next. Logits
do. A logit is the model's raw score for one token as the next token, and a
higher logit means a more likely next token.

GPT-2 has 50,257 tokens, so it writes 50,257 logits, one per token. These are
the five highest after "The capital of France is":

| Token | Logit |
| --- | --- |
| " the" | -100.25 |
| " now" | -100.82 |
| " a" | -100.86 |
| " France" | -101.21 |
| " Paris" | -101.21 |

The logits are all near -100, and that's normal. Only the differences between
them matter, so " the" leads " now" by 0.57.

### The `logits` command

`logits` reads three files:

- `model.safetensors`, with every tensor `embed` and `block` read, plus a final
  layer norm:
    - `ln_f.weight`, with shape `[n_embd]`
    - `ln_f.bias`, with shape `[n_embd]`
- `config.json`, the same format as for `attn`
- `ids.json`, the same format as for `embed`

Compute the logits like this:

```text
rows = the embedding of the ids, as embed does
for each layer from 0 to n_layer - 1:
    rows = the block's output for rows, as block does

// layer norm with the same rule and 0.00001 as ln_1 in a block
last = layer_norm(the last row of rows, ln_f.weight, ln_f.bias)

for each token t:
    logit[t] = dot(last, row t of wte.weight)
```

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
expected number, whichever is larger. It will also run a two-token sequence,
where only the last position is scored, and a second layer that changes the
scores.
