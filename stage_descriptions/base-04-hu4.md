In this stage, you'll implement the `block` command to run one transformer
block.

### Why a block?

GPT-2 stacks 12 of these blocks. Each one runs the attention you built with
`attn`, then a small network called the MLP (multilayer perceptron), and adds
each result onto the row rather than replacing it. The MLP works on each row
alone. It widens the row from 768 numbers to 3072, then brings it back to 768.

Because every block adds onto the row, the row's numbers keep growing. Here is
how spread out the row for "is" is, as its standard deviation, going into
several of GPT-2's layers:

```text
layer     0      3      6      9      11     after 11
spread    0.18   2.09   2.65   4.59   10.46  15.50
```

So before attention and before the MLP, each block layer-normalizes the row.
That brings it back to a spread of about 1, however large its numbers have
grown.

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

Layer norm and GELU work on one row, or one number, at a time:

```text
layer_norm(row, weight, bias):
    mean = the average of the row's numbers
    variance = the average of (x - mean)^2 over the row's numbers
    for each index i:
        x = (row[i] - mean) / sqrt(variance + 0.00001)
        out[i] = x * weight[i] + bias[i]

gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

Run the block like this:

```text
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

Copy constants like 0.00001 and 0.044715 exactly as written. They're part of
GPT-2's definition, so there's nothing to work out about them, and a different
value changes the output.

### Example

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
block whose MLP changes the row to about `[1.841192, 2]`, and layer 1 of a
two-layer file.

### Notes

- Step 3 adds onto the block's input, not onto the normalized rows from step 1.
  Adding onto the normalized rows changes every number in the output.
- The variance divides by the count of numbers in the row, not by one less, as
  some statistics functions do.
- Read the MLP's width from the shape of `mlp.c_fc.weight` rather than computing
  it from `n_embd`.
