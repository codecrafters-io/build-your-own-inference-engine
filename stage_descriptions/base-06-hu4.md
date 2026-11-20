In this stage, you'll finish the `block` command by adding the MLP that follows
attention.

### Why an MLP?

Attention moves information between rows. The other half of a transformer block
does the opposite: it works on each row on its own and never looks at another.
That half is an [MLP](https://www.3blue1brown.com/lessons/mlp), short for
multilayer perceptron, and it holds about two-thirds of a block's numbers.

It's two multiplies by a weight with a function called GELU between them. The
first widens the row from 768 numbers to 3,072, and the second brings it back.
GELU keeps positive numbers near themselves and pushes negative ones toward 0,
so the wide middle has something to learn that two multiplies alone could not.

The MLP is wrapped the way attention is, with a layer norm of its own, so a
block normalizes twice:

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

After the layer norm, attention, and residual add you already run, add three
more steps:

1. Layer-normalize a copy of each row with `ln_2.weight` and `ln_2.bias`.
2. Multiply each copy by `mlp.c_fc.weight`, add `mlp.c_fc.bias`, and apply GELU
   to every number. The row is now `4 * n_embd` numbers wide.
3. Multiply by `mlp.c_proj.weight`, add `mlp.c_proj.bias`, and add the result
   onto the rows those copies came from.

GELU is one formula, and its constants are load-bearing. Some libraries offer an
exact version that this one approximates, and the two differ by up to about
0.0005, which is enough to fail the tester:

```text
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

### One block, step by step

This example continues the one you worked through for the layer norm and the
residual add. The layer norms have weight `[1, 1]` and bias `[0, 0]`, the MLP's
weights keep the first two numbers of the wide row and put them back, and the
input is `[[1, 2]]`:

```text
// the layer norm, attention, and residual add you already run
row           [1, 2] + [-1, 1] = about [0, 3]

// 1. layer norm with ln_2, the same rule
normalized    about [-1, 1]

// 2. × c_fc.weight + c_fc.bias, then GELU on every number
wide row      [-1, 1, 0, 0, 0, 0, 0, 0]     // 4 * n_embd numbers
after GELU    [-0.1588, 0.8412, 0, 0, 0, 0, 0, 0]

// 3. × c_proj.weight + c_proj.bias, added onto the row above
output        [0, 3] + [-0.1588, 0.8412] = [-0.1588, 3.8412]
```

Print one JSON array of rows:

```json
[[-0.15878819, 3.84116958]]
```

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

The tester allows a difference of 0.0001 on each number. It will also run three
rows of four numbers through a block whose weights are all random numbers
between -1 and 1, once for layer 0 and once for layer 1 of a two-layer file.

### Notes

- GPT-2's MLP is always four times the row's width, so `mlp.c_fc.weight` has
  shape `[n_embd, 4 * n_embd]`. Reading the width from that shape costs nothing
  and keeps the code honest if a model ever differs.
