In this stage, you'll implement `block` to normalize rows before attention and
add attention's output back to the original rows.

### Why wrap attention?

The output of `attn` combines transformed information from several positions.
On its own, that output has no direct path carrying the original input row
forward. A residual connection supplies that path by adding the input row to
attention's output, number by number.

A [transformer](https://www.3blue1brown.com/lessons/gpt) combines attention with
per-row calculations in repeated blocks. GPT-2 has 12 blocks with the same
structure and different learned weights. Each block's attention output is an
update added to its input.

Repeated updates can change the scale of a row's numbers. Before attention,
GPT-2 rescales a copy of each row with
[layer normalization](https://www.pinecone.io/learn/batch-layer-normalization/).
It subtracts the row's average and divides by a measure of its spread, then
applies a learned scale and bias. With scale 1 and bias 0, these two rows
normalize to almost the same values:

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
3. Add the attention output onto the block's original input rows. The normalized
   copies are only inputs to attention.

Layer norm uses the formula below. The mean is the average of the row's
numbers. The variance is the average of `(x - mean)^2`, which measures their
spread. The constant 0.00001 prevents division by zero for a constant row and
matches GPT-2's definition. Apply the corresponding weight and bias to each
coordinate:

```text
layer_norm(x) = (x - mean) / sqrt(variance + 0.00001) * weight + bias
```

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

The tester allows a difference of 0.0001 on each number. It will also run three
rows of four numbers through a block whose `ln_1` and attention weights are
random numbers between -1 and 1, once for layer 0 and once for layer 1 of a
two-layer file. Those models have two heads.

### Notes

- Divide by the count of numbers in the row when you take the variance. Many
  built-in variance functions divide by one less, which turns 0.25 into 0.5 for
  the row `[1, 2]` and the output into `[0.29, 2.71]` instead of `[0, 3]`.
- The file also holds `ln_2` and the MLP tensors, which later stages use. They
  are all zeros here, so you can ignore them for now.
