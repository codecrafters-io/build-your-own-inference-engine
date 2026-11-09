In this stage, you'll implement the `block` command to run attention inside a
layer norm and a residual add.

### Why wrap attention?

`attn` replaces each row with a weighted average of itself and the rows before
it, and that throws something away. In the weights you saw when you built
`attn`, 0.57 of the new row for "is" came from "France" and only 0.06 came from
"is". The row has taken on the words before it and lost most of what it held.

A [transformer](https://www.3blue1brown.com/lessons/gpt) keeps both. GPT-2 is a
stack of 12 identical blocks, and a block adds attention's output onto the row
it started from instead of replacing it. The row keeps what it was, plus what
attention found.

Adding onto a row over and over makes its numbers grow. So each block first
rescales a copy of the row, a step called
[layer normalization](https://www.pinecone.io/learn/batch-layer-normalization/).
It shifts and scales the numbers so their average is 0 and their standard
deviation is about 1, then applies a weight and a bias of its own. A row and the
same row ten times larger come out almost the same:

```text
row          [2, 4, 6, 8]                 [20, 40, 60, 80]
normalized   [-1.34, -0.45, 0.45, 1.34]   [-1.34, -0.45, 0.45, 1.34]
```

So attention runs on the normalized copy, and its output is added onto the row
the copy came from. A block wraps a second step the same way, which later stages
add.

### The `block` command

`block` takes the same three files and layer number as `attn`. Along with the
four attention tensors, it uses two more. For layer 0 they are:

| Tensor | Shape |
| --- | --- |
| `h.0.ln_1.weight` | `[n_embd]` |
| `h.0.ln_1.bias` | `[n_embd]` |

Run these steps on the input rows:

```text
row ─┬─► ln_1 ─► attention ─► + ─► output
     │                        ▲
     └────────────────────────┘
```

1. Layer-normalize a copy of each row with `ln_1.weight` and `ln_1.bias`.
2. Run attention on those copies, the same way `attn` does.
3. Add the attention output onto the block's input rows, not onto the copies.

Layer norm is one formula. The mean and the variance are taken over the row's
numbers, and the variance is the average of `(x - mean)^2`:

```text
layer_norm(x) = (x - mean) / sqrt(variance + 0.00001) * weight + bias
```

Copy 0.00001 exactly as written. It's part of GPT-2's definition, and a
different value changes the output.

The file also holds `ln_2` and the MLP tensors, which later stages use. In this
stage's files they are all zeros, so you can ignore them for now.

### One block, step by step

`ln_1` has weight `[1, 1]` and bias `[0, 0]`, the attention tensors copy each
row into its query, key, and value, and the input is `[[1, 2]]`:

```text
// 1. layer norm with ln_1
mean          (1 + 2) / 2 = 1.5
variance      (0.5^2 + 0.5^2) / 2 = 0.25
normalized    [-0.5, 0.5] / sqrt(0.25 + 0.00001) = about [-1, 1]

// 2. attention on that copy
// one row attends only to itself, so it gets its own value back
attention     [-1, 1]

// 3. add onto the block's input
output        [1, 2] + [-1, 1] = [0, 3]
```

Print one JSON array of rows:

```json
[[0.0, 3.0]]
```

### Hints

<details>
<summary>What does this look like as pseudocode?</summary>

```text
layer_norm(row, weight, bias):
    mean = the average of the row's numbers
    variance = the average of (x - mean)^2 over the row's numbers
    for each index i:
        x = (row[i] - mean) / sqrt(variance + 0.00001)
        out[i] = x * weight[i] + bias[i]

normed = layer_norm(each row, ln_1.weight, ln_1.bias)
attended = attention(normed)

for each position i:
    result[i] = row[i] + attended[i]
```

</details>

### Tests

The tester will write the same three files as for `attn`. For this example, the
layer is `0`, `input.json` is `[[1, 2]]`, and `config.json` is:

```json
{ "n_embd": 2, "n_head": 1, "n_layer": 2, "n_positions": 4, "vocab_size": 2 }
```

It will then execute your program like this:

```bash
$ ./your_program.sh block model.safetensors config.json 0 input.json
```

Your program must print:

```json
[[0.0, 3.0]]
```

The tester allows a difference of 0.0001 on each number. It will also run two
rows, where each row is normalized on its own numbers, and layer 1 of a
two-layer file.

### Notes

- The variance divides by the count of numbers in the row, not by one less, as
  some statistics functions do.
- Adding onto the normalized copy instead of the row it came from is the common
  mistake here, and it changes every number in the output.
