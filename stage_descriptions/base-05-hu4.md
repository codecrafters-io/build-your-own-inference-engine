In this stage, you'll finish the `block` command by adding the MLP that follows
attention.

### Why an MLP?

Attention moves information between rows. Nothing so far works on a row by
itself, and that is the other half of a transformer block: an
[MLP](https://www.3blue1brown.com/lessons/mlp), short for multilayer
perceptron, which sees one row at a time and nothing else.

It's two multiplies by a weight with a function called GELU between them. The
first widens the row from 768 numbers to 3,072, and the second brings it back.
Those two weight tables hold about two-thirds of each block's numbers.

The MLP is wrapped the same way attention is. The block layer-normalizes a copy
of the row with `ln_2`, runs the MLP on that copy, and adds the result onto the
row it started from. So a block normalizes twice, with its own weights each
time:

```text
row ─┬─► ln_1 ─► attention ─► + ─┬─► ln_2 ─► MLP ─► + ─► output
     │                        ▲  │                  ▲
     └────────────────────────┘  └──────────────────┘
```

### The `block` command

Along with the tensors you already read, a block uses six more. For layer 0:

| Tensor | Shape |
| --- | --- |
| `h.0.ln_2.weight` | `[n_embd]` |
| `h.0.ln_2.bias` | `[n_embd]` |
| `h.0.mlp.c_fc.weight` | `[n_embd, 4 * n_embd]` |
| `h.0.mlp.c_fc.bias` | `[4 * n_embd]` |
| `h.0.mlp.c_proj.weight` | `[4 * n_embd, n_embd]` |
| `h.0.mlp.c_proj.bias` | `[n_embd]` |

Add three steps after the ones you already run:

4. Layer-normalize a copy of each row from step 3 with `ln_2.weight` and
   `ln_2.bias`.
5. Multiply each copy by `mlp.c_fc.weight`, add `mlp.c_fc.bias`, and apply GELU
   to every number. The row is now `4 * n_embd` numbers wide.
6. Multiply by `mlp.c_proj.weight`, add `mlp.c_proj.bias`, and add the result
   onto the rows from step 3.

GELU is one formula, and copying its constants exactly matters as much as it
did for layer norm:

```text
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

### One block, step by step

This example continues the one from a previous stage. The layer norms have
weight `[1, 1]` and bias `[0, 0]`, the MLP's weights keep the first two numbers
of the wide row and put them back, and the input is `[[1, 2]]`:

```text
// steps 1 to 3, as before
row           [1, 2] + [-1, 1] = [0, 3]

// 4. layer norm with ln_2, the same rule
normalized    about [-1, 1]

// 5. × c_fc.weight + c_fc.bias, then GELU on every number
wide row      [-1, 1, 0, 0, 0, 0, 0, 0]     // 4 * n_embd numbers
after GELU    [-0.1588, 0.8412, 0, 0, 0, 0, 0, 0]

// 6. × c_proj.weight + c_proj.bias, added onto the row from step 3
output        [0, 3] + [-0.1588, 0.8412] = [-0.1588, 3.8412]
```

Print one JSON array of rows:

```json
[[-0.15878819, 3.84116958]]
```

### Hints

<details>
<summary>What do the new steps look like as pseudocode?</summary>

```text
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

for each position i:
    // row[i] is the row after attention was added onto it
    n = layer_norm(row[i], ln_2.weight, ln_2.bias)
    wide = gelu(each number of (n × mlp.c_fc.weight + mlp.c_fc.bias))
    row[i] = row[i] + (wide × mlp.c_proj.weight + mlp.c_proj.bias)
```

</details>

<details>
<summary>What does GELU do?</summary>

GELU keeps positive numbers close to themselves and pushes negative ones toward
0. Without it, steps 5 and 6 would each be a multiply and an add, and two of
those in a row work out to a single multiply and add. The wider middle would add
nothing.

</details>

### Tests

The tester will write the same three files as before, with `input.json` set to
`[[1, 2]]` and the layer set to `0`, then execute your program like this:

```bash
$ ./your_program.sh block model.safetensors config.json 0 input.json
```

Your program must print:

```json
[[-0.15878819, 3.84116958]]
```

The tester allows a difference of 0.0001 on each number. It will also run two
rows, where the MLP works on each row alone, and layer 1 of a two-layer file.

### Notes

- Read the MLP's width from the shape of `mlp.c_fc.weight` rather than computing
  it from `n_embd`.
- The MLP never looks at other rows. Only attention does that.
