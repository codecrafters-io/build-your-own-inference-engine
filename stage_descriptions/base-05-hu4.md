In this stage, you'll finish the `block` command by adding the MLP that follows attention.

### The MLP

So far, a block lets each row take in information from the rows before it. The second half of a block works on each row by itself, without looking at the other rows. It's a small [multilayer perceptron](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)#Feedforward_network), or MLP. The MLP takes a row through three steps in turn:

- It multiplies the row by `mlp.c_fc.weight` and adds `mlp.c_fc.bias`, which widens the row to 4 times as many numbers.
- It applies a function called GELU to every number of the wide row.
- It multiplies the wide row by `mlp.c_proj.weight` and adds `mlp.c_proj.bias`, which brings it back to its original length. This is a different tensor from attention's `attn.c_proj.weight`.

In GPT-2, the MLP widens each row from 768 numbers to 3,072, then brings it back to 768. Its tensors hold about two-thirds of the numbers in each block.

### GELU

[GELU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Gaussian-error_linear_unit_(GELU)) is a formula applied to each number on its own. Here's the version GPT-2 uses, where `tanh` is the [hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_functions), built into most languages like `exp`:

```text
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

It keeps positive numbers close to themselves and pushes negative numbers toward 0:

```text
x          -2       -1       0      1       2
gelu(x)    -0.045   -0.159   0      0.841   1.955
```

For example, take the row for "is" in "The capital of France is". In GPT-2's first layer, its wide row has 3,072 numbers. Here are the first 4, before and after GELU:

```text
wide row      [ 0.254, -1.963, -1.600, -1.931, ...]
after GELU    [ 0.152, -0.049, -0.088, -0.052, ...]
```

91% of the numbers in this wide row are negative, so GELU pulls most of them close to 0.

GELU is what makes the MLP worth having. Without it, the two multiply-and-add steps would collapse into one, the same way multiplying by 2 and then by 3 is just multiplying by 6. GELU bends the numbers in between, so the MLP can do more than a single multiply-and-add could.

### The Whole Block

The MLP is wrapped the same way attention is. The block layer-normalizes a copy of the row with its own `ln_2` tensors, runs the MLP on that copy, and adds the result onto the row it copied. So a block normalizes twice, with different weights each time:

```text
row ─┬─► ln_1 ─► attention ─► + ─┬─► ln_2 ─► MLP ─► + ─► new row
     │                        ▲  │                  ▲
     └────────────────────────┘  └──────────────────┘
```

Along with the tensors `block` already uses, the MLP's half uses six more. For layer 0, they're:

| Tensor | Shape |
| --- | --- |
| `h.0.ln_2.weight` | `[n_embd]` |
| `h.0.ln_2.bias` | `[n_embd]` |
| `h.0.mlp.c_fc.weight` | `[n_embd, 4 * n_embd]` |
| `h.0.mlp.c_fc.bias` | `[4 * n_embd]` |
| `h.0.mlp.c_proj.weight` | `[4 * n_embd, n_embd]` |
| `h.0.mlp.c_proj.bias` | `[n_embd]` |

After the layer norm, attention, and residual connection that `block` already runs, your program must:

4. Layer-normalize a copy of each row from step 3 with `ln_2.weight` and `ln_2.bias`
5. Multiply each copy by `mlp.c_fc.weight`, add `mlp.c_fc.bias`, and apply GELU to every number
6. Multiply the result by `mlp.c_proj.weight`, add `mlp.c_proj.bias`, and add that onto the row from step 3

For example, say `n_embd` is 2, so the wide row has 8 numbers, and:

- The row after step 3 is `[-1, 5]`
- `ln_2.weight` is `[1, 1]`, and `ln_2.bias` is `[0, 0]`
- `mlp.c_fc.weight` copies the row's two numbers into the first two wide numbers, and `mlp.c_proj.weight` puts them back
- Both MLP biases are 0

Here are the steps:

```text
// 4. layer norm of a copy, with ln_2
mean          (-1 + 5) / 2 = 2
variance      ((-1 - 2)^2 + (5 - 2)^2) / 2 = 9
normalized    [-3, 3] / sqrt(9 + 0.00001) = about [-1, 1]

// 5. × mlp.c_fc.weight + mlp.c_fc.bias, then GELU on every number
wide row      [-1, 1, 0, 0, 0, 0, 0, 0]
after GELU    [-0.159, 0.841, 0, 0, 0, 0, 0, 0]

// 6. × mlp.c_proj.weight + mlp.c_proj.bias, added onto the row from step 3
new row       [-1, 5] + [-0.159, 0.841] = [-1.159, 5.841]
```

### Tests

The tester will write a `model.safetensors`, a `config.json`, and an `input.json`, then execute your program like this:

```bash
$ ./your_program.sh block model.safetensors config.json 0 input.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with one row for each row in `input.json`
- Each row has `n_embd` numbers
- Each number matches the row after step 6 for that position, to within 0.0001

It will run a few different files, including one with two rows and one that asks for layer 1 of a two-layer file.

### Notes

- Some libraries compute GELU with a function called `erf` instead of the formula above. The two differ by up to about 0.0005, which is more than the tester allows, so use the formula above.
- The tests from earlier stages still pass with the MLP added. Their files set the MLP's tensors to 0, so it adds nothing there.
