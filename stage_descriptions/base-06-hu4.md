In this stage, you'll finish `block` by adding the per-row computation that
follows attention, called an MLP.

### Why an MLP?

Attention moves information between rows. The other half of a transformer block
transforms each row independently, using the same learned weights at every
position.
That half is an [MLP](https://www.3blue1brown.com/lessons/mlp), short for
multilayer perceptron, and it holds about two-thirds of a block's learned weights.

It has two matrix multiplications with a function called GELU between them.
GPT-2's first multiplication widens a row from 768 numbers to 3,072, and the
second brings it back. GELU acts on each number independently, reducing negative
values toward zero while leaving large positive values close to themselves.

GELU is nonlinear: it can't be expressed as just a multiply and an add. Without
it, the two matrix multiplications and biases could be combined into one, and
the wider middle wouldn't add this new kind of computation.

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

Use GPT-2's tanh approximation of GELU, with these constants. Some libraries
default to a different, exact formula, whose results can differ enough to
exceed this stage's tolerance. Here, `tanh` is the hyperbolic tangent function:

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
row           [1, 2] + [-1, 1] = [0, 3]

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
  shape `[n_embd, 4 * n_embd]`. The model's shape determines the width even
  when the test rows are much smaller than GPT-2's.
