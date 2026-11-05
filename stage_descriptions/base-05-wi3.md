In this stage, you'll implement the `logits` command to run every layer and
print the logits for the last position.

### Why logits?

The blocks change each row, but they don't say which token comes next. Logits
do. A logit is the model's raw score for one token as the next token, and a
higher logit means a more likely next token.

After all 12 blocks, GPT-2 normalizes once more with `ln_f`, then scores the
last position against every token row. The score for token `i` is the dot
product of that position with row `i` of `wte.weight`. The real table has 50,257
rows. There is one copy of it, the same table `embed` reads.

### The `logits` command

`config.json` is the same format as for `attn`, and `ids.json` is the same
format as for `embed`. Run `n_layer` blocks, in order, starting from the
embedding of the ids. Layer-normalize the last row with `ln_f.weight` and
`ln_f.bias`, using the same rule as `ln_1` in a block and the same 0.00001. The
score for each token is the dot product of that row with the token's row in
`wte.weight`.

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

### Notes

- Print the scores for the last position only. A one-token prompt has one
  position, and it is the last.
- The score for token `i` uses row `i` of `wte.weight`.
- Run every layer in `config.json`, from layer 0 through `n_layer - 1`.
