In this stage, you'll implement the `attn` command to mix each token's row with
the rows of the tokens before it.

### Why attention?

After embedding, the row for a token is only that token. For example,
take the sentence "The capital of France is". The row for "is" does not yet
depend on "France". Attention is the step that mixes earlier rows into the
current one. "is" looks at "The", "capital", "of", "France", and "is", and its
output row depends on all five. It does not look at a word that comes after it,
because that word has not been written yet.

GPT-2 doesn't mix a row as one block of 768 numbers. It splits the row into 12
slices of 64 numbers, called heads, and each head does its own mix on its slice,
so different heads can pick up different links between words. The 12 results
are then joined back into one row of 768. `n_head` is 12 in
[GPT-2's config](https://huggingface.co/gpt2/blob/main/config.json).

Inside a head, a score says how much one token should look at another. Each
head divides its scores by 8, the square root of its 64 numbers, so the scores
don't grow too large as the slices get wider.

### The `attn` command

The tester writes four files. `model.safetensors` holds the weights.
`config.json` holds the sizes, using the same names as GPT-2's config:

```json
{ "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }
```

`layer.json` holds the number of the layer to run, `0` in this example.
`input.json` is a JSON array of rows, one row per position. This example's input
is `[[1], [2]]`.

`model.safetensors` holds four attention tensors for each layer. Their names
start with `h.` and the layer number, so for layer 0 they are:

| Tensor | Shape | Used in |
| --- | --- | --- |
| `h.0.attn.c_attn.weight` | `[n_embd, 3 * n_embd]` | step 1, multiplied |
| `h.0.attn.c_attn.bias` | `[3 * n_embd]` | step 1, added |
| `h.0.attn.c_proj.weight` | `[n_embd, n_embd]` | step 6, multiplied |
| `h.0.attn.c_proj.bias` | `[n_embd]` | step 6, added |

For layer 1, the names start with `h.1.` instead. A row of `n_embd` numbers
times `c_attn.weight` gives `3 * n_embd` numbers, which is why its shape lists
the row's length first.

For each position, compute its output row like this:

1. Multiply the position's row by `c_attn.weight` and add `c_attn.bias`. This
   gives `3 * n_embd` numbers. The first `n_embd` are the position's query, the
   next `n_embd` are its key, and the last `n_embd` are its value. The query is
   what this position is looking for, the key is what it offers to a position
   that looks at it, and the value is what it hands over when it's chosen.
2. Split the query, the key, and the value into `n_head` heads. Each head gets
   `n_embd / n_head` numbers, called the head size. Steps 3 to 5 run separately
   for each head.
3. Score this position against itself and every earlier position. A score is
   the dot product of this position's query with the other position's key,
   divided by the square root of the head size. Skip later positions.
4. Softmax the scores: raise `e` to each score, then divide each result by the
   sum of all of them. The results are weights that add up to 1.
5. Multiply each scored position's value by its weight, and add them up. That
   sum is the head's output.
6. Join the heads' outputs back into one row of `n_embd` numbers, multiply it by
   `c_proj.weight`, and add `c_proj.bias`. Each head only saw its own slice, and
   this multiply lets every number in the output row draw on every head.

### Example

In this example's file, the attention tensors are:

| Tensor | Value |
| --- | --- |
| `h.0.attn.c_attn.weight` | `[[1, 1, 1]]` |
| `h.0.attn.c_attn.bias` | `[0, 0, 0]` |
| `h.0.attn.c_proj.weight` | `[[1]]` |
| `h.0.attn.c_proj.bias` | `[0]` |

There is one head of size 1, so step 3 divides by 1. Step 1 copies each input
number into the query, the key, and the value:

| Position | Input | Query | Key | Value |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 2 | 2 | 2 | 2 |

Position 0 can only use itself. Its one score is its query times its key,
`1 * 1 = 1`. A single score always gets a weight of 1, so the output is
position 0's value, `1`.

Position 1 uses positions 0 and 1. Its query is 2, so its scores are
`2 * 1 = 2` and `2 * 2 = 4`. Softmax turns them into weights of about 0.119 and
0.881. The output is `0.119 * 1 + 0.881 * 2`, about 1.881.

`c_proj.weight` is 1 and `c_proj.bias` is 0, so step 6 leaves both outputs
unchanged.

Print one JSON array of rows:

```json
[[1.0], [1.8807970779778822]]
```

### Tests

The tester will write the four files and execute your program like this:

```bash
$ ./your_program.sh attn model.safetensors config.json layer.json input.json
```

Your program must print:

```json
[[1.0], [1.8807970779778822]]
```

The tester allows a difference of 0.0001 on each number. It will also run a head
of size 2, where forgetting the division changes the second row, a model with
two heads, and layer 1.

### Notes

- With GPT-2's real weights, scores can be large enough that raising `e` to them
  overflows. Subtract the largest score from every score first. The weights come
  out the same, because softmax only depends on the differences between scores.
