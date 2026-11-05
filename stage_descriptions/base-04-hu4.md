In this stage, you'll implement the `block` command to run one transformer
block.

### Why a block?

GPT-2 stacks 12 of these blocks. Each one normalizes its input, runs the
attention you built with `attn`, and adds that result back onto the input. It
normalizes again, runs a small network called the MLP (multilayer perceptron),
and adds that back too. The MLP widens a row of 768 numbers to 3072 and then
back to 768. `h.0.mlp.c_fc.weight` has shape `[768, 3072]` in the real file.

### The `block` command

The tester writes the same four files as for `attn`: `model.safetensors`,
`config.json`, `layer.json`, and `input.json`. This example's `config.json` is:

```json
{ "n_embd": 2, "n_head": 1, "n_layer": 2, "n_positions": 4, "vocab_size": 2 }
```

`layer.json` is `0`, and `input.json` is `[[1, 2]]`.

Besides the four attention tensors `attn` reads, a block uses eight more. For
layer 0 they are:

| Tensor | Shape | Used in |
| --- | --- | --- |
| `h.0.ln_1.weight` | `[n_embd]` | step 1, multiplied |
| `h.0.ln_1.bias` | `[n_embd]` | step 1, added |
| `h.0.ln_2.weight` | `[n_embd]` | step 4, multiplied |
| `h.0.ln_2.bias` | `[n_embd]` | step 4, added |
| `h.0.mlp.c_fc.weight` | `[n_embd, 4 * n_embd]` | step 5, multiplied |
| `h.0.mlp.c_fc.bias` | `[4 * n_embd]` | step 5, added |
| `h.0.mlp.c_proj.weight` | `[4 * n_embd, n_embd]` | step 6, multiplied |
| `h.0.mlp.c_proj.bias` | `[n_embd]` | step 6, added |

Run these steps on the input rows:

1. Layer-normalize each row with `ln_1`. Find the row's mean, and its variance,
   the average of the squared distances from the mean. Subtract the mean from
   every number, then divide by the square root of the variance plus 0.00001.
   Multiply number by number with `ln_1.weight`, and add `ln_1.bias`. This keeps
   every row at a steady size, however large its numbers have grown.
2. Run attention on the normalized rows, the same way `attn` does.
3. Add the attention output onto the block's input rows, number by number.
   Adding, rather than replacing, lets each row keep what it had and take on
   what attention found.
4. Layer-normalize the rows from step 3 with `ln_2`, using the same rule.
5. Multiply each row by `mlp.c_fc.weight` and add `mlp.c_fc.bias`, which widens
   it to `4 * n_embd` numbers. Then apply GELU to every number:
   `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`. GELU keeps
   positive numbers close to themselves and pushes negative ones toward 0.
   Without it, steps 5 and 6 would collapse into one multiply and one add.
6. Multiply by `mlp.c_proj.weight` and add `mlp.c_proj.bias`, which brings the
   row back to `n_embd` numbers. Add the result onto the rows from step 3.

Attention moves information between rows. The MLP in steps 5 and 6 then works on
each row alone, on what that row now holds.

### Example

In this example's file, both layer norms have weight `[1, 1]` and bias `[0, 0]`,
and every attention and MLP tensor is all zeros. Here is the row after each
step:

| After step | Row |
| --- | --- |
| Input | `[1, 2]` |
| 1. `ln_1` | about `[-1, 1]` |
| 2. Attention | `[0, 0]` |
| 3. Add onto the input | `[1, 2]` |
| 4. `ln_2` | about `[-1, 1]` |
| 5. `c_fc` and GELU | eight zeros |
| 6. `c_proj`, added onto step 3 | `[1, 2]` |

In step 1, the mean of `[1, 2]` is 1.5, and both numbers are 0.5 away from it,
so the variance is 0.25. The square root of 0.25001 is just over 0.5, so the
row becomes about `[-0.5 / 0.5, 0.5 / 0.5]`, which is `[-1, 1]`.

Attention and the MLP both output zeros here, so the two additions add nothing,
and the output is the input:

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

- Step 3 adds onto the block's input, not onto the normalized rows from step 1.
  Adding onto the normalized rows changes every number in the output.
- The variance divides by the count of numbers in the row, not by one less, as
  some statistics functions do.
- Read the MLP's width from the shape of `mlp.c_fc.weight` rather than computing
  it from `n_embd`.
