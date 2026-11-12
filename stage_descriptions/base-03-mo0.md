In this stage, you'll implement the `attn` command to mix each token's row with the rows before it.

### Attention

In earlier stages, you turned each token of a prompt into an [embedding](https://en.wikipedia.org/wiki/Word_embedding), a row of numbers that describes the token and its position. Each row only describes its own token. For example, in "The capital of France is", the row for "is" says nothing about "France", so nothing in it hints at what comes next.

[Attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) is the step that lets a row take in information from other rows. It replaces each row with a new one, built from a weighted mix of itself and the rows before it. GPT-2 runs attention 12 times, one after another, in 12 layers.

For example, when one of GPT-2's later layers builds the new row for "is", it gives the rows of the prompt these shares, called attention weights:

| Token | Attention weight |
| --- | --- |
| "The" | 0.347 |
| "capital" | 0.017 |
| "of" | 0.007 |
| "France" | 0.571 |
| "is" | 0.058 |

More than half of the mix comes from "France".

Despite the name, attention weights aren't among the [weights](https://developers.google.com/machine-learning/glossary#weight) GPT-2 learned in training, so they're not in the weights file. GPT-2 computes them for each prompt, from the rows themselves.

A row only mixes in itself and the rows before it, never the rows after it, even when they're there. GPT-2 writes text one token at a time, from left to right, so it learned to predict each token from the tokens before it. Attention that only looks back like this is called causal attention.

### Queries, Keys, and Values

To compute the attention weights, attention turns each row into three new lists of numbers:

- The query, which describes what this row is looking for in earlier rows
- The key, which describes what this row offers to later rows
- The value, which is what this row passes on to the rows that pick it

For example, the row for "is" might look for "an earlier word that names a place", and the row for "France" might offer "a word that names a place". The two match, so the new row for "is" takes a large share of the value of "France".

GPT-2 computes all three from the row in two steps. First, it multiplies the row by a [tensor](https://app.codecrafters.io/concepts/tensors) of weights called `c_attn.weight`, as the next section explains. Then it adds a [bias](https://developers.google.com/machine-learning/glossary#bias-math-or-bias-term) called `c_attn.bias`, a list of numbers that's added number by number. Each layer has its own copy of both, like `h.0.attn.c_attn.weight` for layer 0.

### Matrix Multiplication

Multiplying a row of `n` numbers by a tensor with shape `[n, m]` is called [matrix multiplication](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/a/multiplying-matrices), and it gives a new row of `m` numbers. Number `j` of the new row is the [dot product](https://en.wikipedia.org/wiki/Dot_product) of the row and column `j` of the tensor: multiply the numbers in matching positions, then add up the products.

For example, here's the row `[1, 2]` multiplied by a tensor with shape `[2, 3]`:

```text
tensor        column 0   column 1   column 2
row 0             1          0          2
row 1             3          1          0

[1, 2] × tensor = [1×1 + 2×3,  1×0 + 2×1,  1×2 + 2×0]
                = [7, 2, 2]
```

The starter code's loader gives you this tensor as the flat list `[1, 0, 2, 3, 1, 0]`. The number in row `i`, column `j` is at index `i * m + j`. For example, row 1, column 0 is at index `1 * 3 + 0 = 3`.

In GPT-2, the row for "is" becomes its query, key, and value like this:

```text
row for "is"         768 numbers
× c_attn.weight      shape [768, 2304]
+ c_attn.bias        2304 numbers
= 2304 numbers       [ query: 768 numbers | key: 768 numbers | value: 768 numbers ]
```

### Heads

GPT-2 doesn't compute just one set of attention weights for each row. It splits the query, the key, and the value into 12 slices of 64 numbers, and runs the rest of attention on each set of slices separately, side by side. Each of these is called a head, so each of GPT-2's 12 layers has 12 heads. Each head computes its own attention weights, so it can look for something different:

```text
query for "is"   [ numbers 0–63 | numbers 64–127 | ... | numbers 704–767 ]
                      head 0          head 1                 head 11
```

The key and the value are split the same way. Inside a head, the query, the key, and the value mean just the head's slices of them. The length of a slice, 64 in GPT-2, is called the head size.

### Scores and Softmax

To decide how much of each row to mix in, a head gives each row a score: the dot product of the current row's query and that row's key, divided by the square root of the head size. Dividing keeps the scores small, since longer slices give larger dot products. GPT-2's head size is 64, so it divides by 8.

[Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/) then turns the head's scores into attention weights. It raises `e` to the power of each score (`exp` in most languages) and divides each result by their total, so the attention weights are all positive and add up to 1. For example, softmax turns the scores `[0, 0]` into the attention weights `[0.5, 0.5]`, and `[0, 1]` into `[0.27, 0.73]`.

The head then adds up the values of those rows, number by number, each multiplied by its attention weight. The result is as long as one slice.

The table at the start of this stage came from head 8 in layer 9. Here's how that head scores the rows of the prompt when it builds the new row for "is":

```text
                  "The"   "capital"    "of"    "France"    "is"
dot product      12.152    -12.048   -18.772    16.132   -2.105
÷ 8 = score       1.519     -1.506    -2.346     2.017   -0.263
softmax           0.347      0.017     0.007     0.571    0.058
```

"France" has the highest score, so it gets the biggest attention weight.

Once every head is done, their results go side by side, in order, to make one row of 768 numbers. Last, attention multiplies that row by one more tensor, `c_proj.weight`, and adds `c_proj.bias`, which mixes the heads' results together. The result is the token's new row.

### The `attn` Command

`attn` runs the attention of one layer. It takes four arguments:

1. `model.safetensors`, a weights file with the attention tensors of every layer
2. `config.json`, a JSON object of sizes, like `{"n_embd": 768, "n_head": 12, ...}`
3. A layer number, like `0`. `attn` runs only that layer's attention.
4. `input.json`, a JSON array of rows, one for each position, like `[[1], [0]]`

`n_embd` is how many numbers each row has, and `n_head` is the number of heads, so the head size is `n_embd / n_head`. These are the names [GPT-2's own config](https://huggingface.co/openai-community/gpt2/blob/main/config.json) uses, and you only need these two fields for now.

Each layer has its own four attention tensors, whose names start with `h.` and the layer number. For layer 0, they're:

| Tensor | Shape |
| --- | --- |
| `h.0.attn.c_attn.weight` | `[n_embd, 3 * n_embd]` |
| `h.0.attn.c_attn.bias` | `[3 * n_embd]` |
| `h.0.attn.c_proj.weight` | `[n_embd, n_embd]` |
| `h.0.attn.c_proj.bias` | `[n_embd]` |

`attn` prints a JSON array of new rows, one for each input row, each with `n_embd` numbers. Your program must:

1. Multiply every row by `c_attn.weight`, add `c_attn.bias`, and split the result into the row's query, key, and value

Then, for the row at each position `i`, in each head, using the head's slices:

2. Score position `i` against itself and each earlier position: the dot product of `i`'s query and that position's key, divided by the square root of the head size
3. Softmax the scores into attention weights
4. Add up the values of those positions, number by number, each multiplied by its attention weight

Finally, for each position:

5. Put the heads' results side by side, in order, multiply by `c_proj.weight`, and add `c_proj.bias`

For example, say each row is one number, so `n_embd` and `n_head` are both `1`, and the input is `[[1], [0]]`. The tensors are:

- `c_attn.weight` is `[[2, 1, 3]]`, so the query is 2 times the row, the key is 1 times the row, and the value is 3 times the row
- `c_attn.bias` is `[0, 0, 1]`, which adds 1 to the value
- `c_proj.weight` is `[[2]]`, and `c_proj.bias` is `[1]`

Here are the steps, with each one's results under its number:

```text
// 1. row × c_attn.weight + c_attn.bias
position 0    [1] → query 2, key 1, value 3 + 1 = 4
position 1    [0] → query 0, key 0, value 0 + 1 = 1

// 2. dot product of query and key, ÷ √1, for itself and earlier positions
position 0    2 × 1 = 2
position 1    0 × 1 = 0,  0 × 0 = 0

// 3. softmax
position 0    [1]           // a single score always gets weight 1
position 1    [0.5, 0.5]    // equal scores get equal weights

// 4. the values, each times its attention weight, added up
position 0    1 × 4 = 4
position 1    0.5 × 4 + 0.5 × 1 = 2.5

// 5. × c_proj.weight + c_proj.bias
position 0    4 × 2 + 1 = 9
position 1    2.5 × 2 + 1 = 6
```

So `attn` prints `[[9.0], [6.0]]`.

With two heads, say `n_embd` and `n_head` are both `2`, so each head gets one number. The input is `[[1, 0], [0, 1]]`. To keep it short, say `c_attn` copies each row into its query, key, and value unchanged, and `c_proj` leaves the result as it is. Here's position 1:

```text
head 0 (number 0)   scores   0 × 1 = 0,  0 × 0 = 0     weights  [0.5, 0.5]
                    value    0.5 × 1 + 0.5 × 0 = 0.5
head 1 (number 1)   scores   1 × 0 = 0,  1 × 1 = 1     weights  [0.269, 0.731]
                    value    0.269 × 0 + 0.731 × 1 = 0.731
side by side        [0.5, 0.731]
```

### Tests

The tester will write a `model.safetensors`, a `config.json`, and an `input.json`, then execute your program like this:

```bash
$ ./your_program.sh attn model.safetensors config.json 0 input.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with one row for each row in `input.json`
- Each row has `n_embd` numbers
- Each number matches the attention output for that position, to within 0.0001

The tester reads your output as JSON numbers, so `9` and `9.0` both match. It will run a few different files, including one with two heads and one that asks for layer 1 of a two-layer file.

### Notes

- Steps 1 and 5 both multiply a row by a tensor, and later stages do it again, so it's worth writing as its own function.
- To see attention on your own sentences, the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) runs GPT-2 in your browser and draws each head's attention weights.
