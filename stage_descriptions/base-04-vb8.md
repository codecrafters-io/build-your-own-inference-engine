In this stage, you'll implement the `block` command, which wraps attention in a layer norm and a residual connection.

### Transformer Blocks

In earlier stages, you implemented attention, which GPT-2 runs once in each of its 12 layers. A layer does more than attention, though. Each of GPT-2's layers is called a transformer block, after the [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), the design GPT-2 is built on. A block has two halves:

1. Attention, which mixes each row with the rows before it
2. An MLP, short for multilayer perceptron, which works on each row by itself

This stage builds the first half, where two more steps go around attention: a residual connection and a layer norm. Later stages add the MLP.

### Residual Connections

`attn` replaces each row with a new one, but a block doesn't. It adds attention's output onto the row it started from, number by number. This is called a [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network#Residual_connection).

That way, a row keeps everything it already held, like which token it is and where it sits, and attention only adds what it found. Each of the 12 layers adds more onto the same rows.

### Layer Normalization

Adding onto a row layer after layer makes its numbers spread out more and more. For example, take GPT-2's row for "is" in "The capital of France is". Here's how spread out its numbers are as it goes through the layers, measured by their [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation):

```text
row for "is"         standard deviation
entering layer 0           0.18
entering layer 1           2.04
entering layer 6           2.65
entering layer 11         10.46
```

So before attention, each block rescales a copy of the row. This step is called [layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization), or layer norm. It shifts and scales the copy's numbers so that their mean is 0 and their standard deviation is about 1. That way, the copy attention gets is about as spread out in every layer. Then layer norm multiplies each number by the weight at its position and adds the bias at its position. The weight and the bias are lists with one number for each number in the row.

Layer norm is one formula. The mean is the average of the row's numbers, the variance is the average of `(x - mean)^2` over them, and the variance's square root is the standard deviation:

```text
layer_norm(x) = (x - mean) / sqrt(variance + 0.00001) * weight + bias
```

The `0.00001` is part of GPT-2's definition. It keeps the division safe when all of a row's numbers are equal.

For example, here's a row and the same row ten times larger, with a weight of 1 and a bias of 0. Both come out the same:

```text
row          [2, 4, 6, 8]                 [20, 40, 60, 80]
mean         5                            50
variance     (9 + 1 + 1 + 9) / 4 = 5      (900 + 100 + 100 + 900) / 4 = 500
normalized   [-1.34, -0.45, 0.45, 1.34]   [-1.34, -0.45, 0.45, 1.34]
```

### The `block` Command

`block` takes the same four arguments as `attn`, and runs the first half of one layer's block. Along with the four attention tensors, it uses two more. For layer 0, they're:

| Tensor | Shape |
| --- | --- |
| `h.0.ln_1.weight` | `[n_embd]` |
| `h.0.ln_1.bias` | `[n_embd]` |

`block` prints a JSON array of new rows, one for each input row, each with `n_embd` numbers. Your program must:

1. Layer-normalize a copy of each row with `ln_1.weight` and `ln_1.bias`, using that row's own mean and variance
2. Run attention on the copies, the same way `attn` does
3. Add attention's output for each position onto the original row, not onto the copy

```text
row ─┬─► ln_1 ─► attention ─► + ─► new row
     │                        ▲
     └────────────────────────┘
```

For example, here's the first half of GPT-2's first layer on the row for "The", in "The capital of France is", with the first 4 of its 768 numbers:

```text
row for "The"           [-0.09, -0.22,  0.07, -0.05, ...]
attention's output    + [ 0.21, -0.21,  0.74, -0.01, ...]
                        ---------------------------------
new row for "The"     = [ 0.12, -0.43,  0.81, -0.06, ...]
```

To check your program by hand, say `n_embd` is 2, `n_head` is 1, and the input is `[[1, 3]]`. The tensors are:

- `ln_1.weight` is `[2, 1]`, and `ln_1.bias` is `[0, 1]`
- `c_attn.weight` is `[[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]` and `c_proj.weight` is `[[1, 0], [0, 1]]`, with biases of 0. They pass each row through unchanged, so attention on a single row returns that row

Here are the steps:

```text
// 1. layer norm of a copy, with ln_1
mean             (1 + 3) / 2 = 2
variance         ((1 - 2)^2 + (3 - 2)^2) / 2 = 1
normalized       [-1, 1] / sqrt(1 + 0.00001) = about [-1, 1]
× weight + bias  [-1 × 2 + 0, 1 × 1 + 1] = [-2, 2]

// 2. attention on the copy
attention        [-2, 2]

// 3. add onto the original row
new row          [1, 3] + [-2, 2] = [-1, 5]
```

So `block` prints about `[[-1.0, 5.0]]`.

### Tests

The tester will write a `model.safetensors`, a `config.json`, and an `input.json`, then execute your program like this:

```bash
$ ./your_program.sh block model.safetensors config.json 0 input.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with one row for each row in `input.json`
- Each row has `n_embd` numbers
- Each number matches the row after step 3 for that position, to within 0.0001

It will run a few different files, including one with two rows and one that asks for layer 1 of a two-layer file.

### Notes

- The variance divides by the number of values in the row. Some statistics functions divide by one less instead.{{#lang_is_python}} Python's `statistics.variance` divides by one less, so use `statistics.pvariance` or compute it yourself.{{/lang_is_python}}
- The weights file also holds `ln_2` and the MLP's tensors. Later stages use them, so you can ignore them for now.
