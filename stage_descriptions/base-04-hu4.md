In this stage, you'll implement the `block` command to run one transformer
block.

### Why a block?

GPT-2 stacks 12 of these blocks. Each one normalizes its input, runs the
attention you built with `attn`, and adds that result back onto the input. It
normalizes again, runs a small network called the MLP (multilayer perceptron),
and adds that back too. The MLP widens a row of 768 numbers to 3072 and then
back to 768. `h.0.mlp.c_fc.weight` has shape `[768, 3072]` in the real file.

### The `block` command

`config.json`, `layer.json`, and `input.json` are the same format as for
`attn`. This example's input is `[[1, 2]]`, and `layer.json` is `0`.

A block reads eight more tensors, all under `h.<layer>`: `ln_1.weight`,
`ln_1.bias`, `ln_2.weight`, `ln_2.bias`, `mlp.c_fc.weight`, `mlp.c_fc.bias`,
`mlp.c_proj.weight`, and `mlp.c_proj.bias`. The attention tensors are the same
ones `attn` reads.

1. Layer-normalize each row with `ln_1`. The mean is the average of the row. The
   variance is the average of the squared distances from that mean. Subtract the
   mean from every number, then divide by the square root of the variance plus
   0.00001. Multiply by `ln_1.weight` and add `ln_1.bias`.
2. Run attention on those rows.
3. Add the attention output onto the original rows.
4. Layer-normalize those sums with `ln_2`, using the same rule and 0.00001.
5. Multiply by `mlp.c_fc.weight`, add `mlp.c_fc.bias`, and apply GELU. For a
   number `x`, GELU is
   `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`. GPT-2's config
   names this `gelu_new`.
6. Multiply by `mlp.c_proj.weight`, add `mlp.c_proj.bias`, and add that onto the
   rows from step 3.

In this file the attention weights and the MLP weights are zero, so both
additions add nothing. The output is the input.

```json
[[1.0, 2.0]]
```

### Tests

The tester will write the four files and execute your program like this:

```bash
$ ./your_program.sh block model.safetensors config.json layer.json input.json
```

Your program must print:

```json
[[1.0, 2.0]]
```

The tester allows a difference of 0.0001 on each number. It will also run a
block whose MLP changes the row to about `[1.841192, 2]`, and layer 1 of a
two-layer file.

### Notes

- Add the attention output onto the block's input, then add the MLP output onto
  that sum.
- The MLP's hidden row is 4 times `n_embd`. Read the width from
  `mlp.c_fc.weight`.
- The layer-norm epsilon is 0.00001, the value in GPT-2's config.
