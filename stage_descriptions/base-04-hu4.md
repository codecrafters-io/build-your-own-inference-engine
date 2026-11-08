In this stage, you'll implement the `block` command to run one transformer
block.

### Why a block?

GPT-2 is a transformer, a model built as a stack of identical blocks, and it
has 12 of them. A block does two things to each row. It runs the attention you
built with `attn`, then the MLP, short for multilayer perceptron. Each result is
added onto the row instead of replacing it.

Attention moves information between rows. The MLP then works on each row alone,
on what that row now holds. It's two multiplies by a weight, with a function
called GELU between them. The first widens the row from 768 numbers to 3,072,
and the second brings it back to 768. These two weight tables hold about
two-thirds of each block's numbers.

### Layer normalization

Because every block adds onto the row, its numbers keep growing. So each block
first rescales a copy of the row, a step called layer normalization. It shifts
and scales the numbers so their average is 0 and their standard deviation is 1,
then applies a weight and a bias of its own. For example, a row and the same row
ten times larger come out the same, before the weight and bias:

```text
row          [2, 4, 6, 8]                 [20, 40, 60, 80]
normalized   [-1.34, -0.45, 0.45, 1.34]   [-1.34, -0.45, 0.45, 1.34]
```

Attention and the MLP work on that copy, and their results are added onto the
original row.

### The `block` command

`block` takes the same three files and layer number as `attn`. Besides the four
attention tensors, a block uses eight more. For layer 0 they are:

| Tensor | Shape |
| --- | --- |
| `h.0.ln_1.weight` | `[n_embd]` |
| `h.0.ln_1.bias` | `[n_embd]` |
| `h.0.ln_2.weight` | `[n_embd]` |
| `h.0.ln_2.bias` | `[n_embd]` |
| `h.0.mlp.c_fc.weight` | `[n_embd, 4 * n_embd]` |
| `h.0.mlp.c_fc.bias` | `[4 * n_embd]` |
| `h.0.mlp.c_proj.weight` | `[4 * n_embd, n_embd]` |
| `h.0.mlp.c_proj.bias` | `[n_embd]` |

A block runs these steps on the input rows. The two lines under the diagram are
the additions in steps 3 and 6. Each one adds onto the row from before the layer
norm, not onto the normalized copy:

```text
row ─┬─► ln_1 ─► attention ─► + ─┬─► ln_2 ─► MLP ─► + ─► output
     │                        ▲  │                  ▲
     └────────────────────────┘  └──────────────────┘
```

1. Layer-normalize a copy of each row with `ln_1.weight` and `ln_1.bias`.
2. Run attention on those copies, the same way `attn` does.
3. Add the attention output onto the block's input rows.
4. Layer-normalize a copy of each row from step 3 with `ln_2.weight` and
   `ln_2.bias`.
5. Multiply each copy by `mlp.c_fc.weight`, add `mlp.c_fc.bias`, and apply GELU
   to every number. The row is now `4 * n_embd` numbers wide.
6. Multiply by `mlp.c_proj.weight`, add `mlp.c_proj.bias`, and add the result
   onto the rows from step 3.

Layer norm and GELU are each one formula. In layer norm, the mean and the
variance are taken over the row's numbers, and the variance is the average of
`(x - mean)^2`:

```text
layer_norm(x) = (x - mean) / sqrt(variance + 0.00001) * weight + bias
gelu(x)       = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

Copy constants like 0.00001 and 0.044715 exactly as written. They're part of
GPT-2's definition, so there's nothing to work out about them, and a different
value changes the output.

### One block, step by step

In this example's file, both layer norms have weight `[1, 1]` and bias `[0, 0]`,
and every attention and MLP tensor is all zeros. The input is `[[1, 2]]`:

```text
// 1. layer norm with ln_1
mean          (1 + 2) / 2 = 1.5
variance      (0.5^2 + 0.5^2) / 2 = 0.25
normalized    [-0.5, 0.5] / sqrt(0.25 + 0.00001) = about [-1, 1]
weight, bias  [-1, 1] * [1, 1] + [0, 0] = [-1, 1]

// 2. attention, with every tensor zero
attention     [0, 0]

// 3. add onto the block's input
row           [1, 2] + [0, 0] = [1, 2]

// 4. layer norm with ln_2, the same rule
normalized    about [-1, 1]

// 5. × c_fc.weight + c_fc.bias, then GELU
wide row      [0, 0, 0, 0, 0, 0, 0, 0]       // 4 * n_embd numbers

// 6. × c_proj.weight + c_proj.bias, added onto the row from step 3
output        [1, 2] + [0, 0] = [1, 2]
```

Print one JSON array of rows:

```json
[[1.0, 2.0]]
```

### Hints

<details>
<summary>What does the whole block look like as pseudocode?</summary>

```text
layer_norm(row, weight, bias):
    mean = the average of the row's numbers
    variance = the average of (x - mean)^2 over the row's numbers
    for each index i:
        x = (row[i] - mean) / sqrt(variance + 0.00001)
        out[i] = x * weight[i] + bias[i]

gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

// 1. layer norm with ln_1
normed = layer_norm(each row, ln_1.weight, ln_1.bias)

// 2. attention on the normalized rows, as attn does
attended = attention(normed)

for each position i:
    // 3. add onto the block's input
    row[i] = row[i] + attended[i]

    // 4. layer norm with ln_2
    n = layer_norm(row[i], ln_2.weight, ln_2.bias)

    // 5. widen to 4 * n_embd numbers, then GELU each one
    wide = gelu(each number of (n × mlp.c_fc.weight + mlp.c_fc.bias))

    // 6. back to n_embd numbers, added onto the row from step 3
    row[i] = row[i] + (wide × mlp.c_proj.weight + mlp.c_proj.bias)
```

</details>

<details>
<summary>Why add the results onto the input instead of replacing it?</summary>

Adding lets each row keep what it had and take on what attention or the MLP
found. Each of GPT-2's 12 blocks adds a little to the row, rather than rewriting
it.

</details>

<details>
<summary>What does GELU do?</summary>

GELU keeps positive numbers close to themselves and pushes negative ones toward
0. Without it, steps 5 and 6 would each be a multiply and an add, and two of
those in a row work out to a single multiply and add. The wider middle would
add nothing.

</details>

### Tests

The tester will write the same three files as for `attn`. For this example,
the layer is `0`, `input.json` is `[[1, 2]]`, and `config.json` is:

```json
{ "n_embd": 2, "n_head": 1, "n_layer": 2, "n_positions": 4, "vocab_size": 2 }
```

It will then execute your program like this:

```bash
$ ./your_program.sh block model.safetensors config.json 0 input.json
```

Your program must print:

```json
[[1.0, 2.0]]
```

The tester allows a difference of 0.0001 on each number. It will also run a
block whose MLP changes the row, and layer 1 of a two-layer file.

### Notes

- The variance divides by the count of numbers in the row, not by one less, as
  some statistics functions do.
- Read the MLP's width from the shape of `mlp.c_fc.weight` rather than computing
  it from `n_embd`.
